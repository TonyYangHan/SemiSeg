import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
import torchvision.transforms.functional as TF
from tqdm import tqdm

# 1. Augmentation Helper
# Applies random brightness/contrast, flips, and rotation
# img: CxHxW tensor [0,1]; mask: 1xHxW tensor
def augment_image(img, mask=None):
    if random.random() < 0.5:
        img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
    if random.random() < 0.5:
        img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
    if random.random() < 0.5:
        img = TF.hflip(img)
        if mask is not None:
            mask = TF.hflip(mask)
    if random.random() < 0.5:
        img = TF.vflip(img)
        if mask is not None:
            mask = TF.vflip(mask)
    angle = random.uniform(-15, 15)
    img = TF.rotate(img, angle)
    if mask is not None:
        mask = TF.rotate(mask, angle)
    return (img, mask) if mask is not None else img

# 2. Dataset Definition
class BrightfieldDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, resize=None):
        self.images = image_paths
        self.masks = mask_paths
        self.labeled = mask_paths is not None
        self.resize = resize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.resize:
            img = img.resize(self.resize, Image.BILINEAR)
        img = TF.to_tensor(img)

        if self.labeled:
            m = Image.open(self.masks[idx]).convert('L')
            if self.resize:
                m = m.resize(self.resize, Image.NEAREST)
            raw = np.array(m)
            mask = torch.from_numpy((raw > 127).astype(np.float32)).unsqueeze(0)
            img, mask = augment_image(img, mask)
            mask = (mask > 0.5).float()
            return img, mask

        img = augment_image(img)
        return img

# 3. Multi-Scale Blocks and U-Net with checkpointing
class MultiScaleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 1)
        self.c3 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.c5 = nn.Conv2d(in_ch, out_ch, 5, padding=2)
        self.c7 = nn.Conv2d(in_ch, out_ch, 7, padding=3)
        self.fuse = nn.Conv2d(out_ch*4, out_ch, 1)
        # self.dropout = nn.Dropout2d(0.02)

    def forward(self, x):
        y = torch.cat([
            F.elu(self.c1(x)),
            F.elu(self.c3(x)),
            F.elu(self.c5(x)),
            F.elu(self.c7(x))
        ], dim=1)
        y = F.elu(self.fuse(y))
        # y = self.dropout(y)
        return y

class MultiScaleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = MultiScaleBlock(3,64)
        self.e2 = MultiScaleBlock(64,128)
        self.e3 = MultiScaleBlock(128,256)
        self.e4 = MultiScaleBlock(256,512)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d3 = MultiScaleBlock(256+512,256)
        self.d2 = MultiScaleBlock(128+256,128)
        self.d1 = MultiScaleBlock(64+128,64)
        self.out = nn.Conv2d(64,1,1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = checkpoint(self.e3, self.pool(x2), use_reentrant=False)
        x4 = checkpoint(self.e4, self.pool(x3), use_reentrant=False)
        y = self.up(x4)
        y = checkpoint(self.d3, torch.cat([y,x3],1), use_reentrant=False)
        y = self.up(y)
        y = checkpoint(self.d2, torch.cat([y,x2],1), use_reentrant=False)
        y = self.up(y)
        y = checkpoint(self.d1, torch.cat([y,x1],1), use_reentrant=False)
        return self.out(y)

# 4. Main Training Script
def main():
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    # File paths\    
    labeled_imgs   = sorted(glob('selected_images_1/*.jpg'))
    mask_imgs      = sorted(glob('annotation_1/*.jpg'))
    unlabeled_imgs = sorted(glob('pool_images/*.jpg'))

    # Datasets & Loaders
    resize = (512,512)
    sup_ds   = BrightfieldDataset(labeled_imgs, mask_imgs, resize)
    unsup_ds = BrightfieldDataset(unlabeled_imgs, None, resize)
    num_workers = max(1, os.cpu_count()-1)
    sup_loader   = DataLoader(sup_ds, batch_size=4, shuffle=True,  num_workers=num_workers, pin_memory=True)
    unsup_loader = DataLoader(unsup_ds, batch_size=4, shuffle=True,  num_workers=num_workers, pin_memory=True)

    # Device & Weight Calc
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_pos, all_neg = 0,0
    for p in mask_imgs:
        arr = np.array(Image.open(p).convert('L'))>127
        all_pos += arr.sum(); all_neg += (~arr).sum()
    pos_weight = all_neg/(all_pos+1e-6)
    neg_weight = 1.0
    print(f"Using pos_weight={pos_weight:.3f}")


    # Models, Optimizer, Scheduler
    student = MultiScaleUNet().to(device)
    teacher = MultiScaleUNet().to(device)
    teacher.load_state_dict(student.state_dict())
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience = 20)
    scaler = torch.amp.GradScaler('cuda')
    ema_decay, base_cons_w, epochs = 0.99, 0.1, 500
    rampup_epochs = 10; accumulate_steps = 4; patience= 30; no_imp=0; best_loss=float('inf')
    save_name = 'best_student_v2.pth'


    # Training Loop
    for ep in range(1, epochs+1):

        cons_w = base_cons_w*min(1,ep/rampup_epochs)
        student.train()
        tot_loss,count=0.0,0
        optimizer.zero_grad()
        loop = tqdm(zip(sup_loader,unsup_loader), total=min(len(sup_loader),len(unsup_loader)), desc=f"Epoch {ep}/{epochs}")


        for (imgs,masks), imgs_u in loop:
            imgs,masks,imgs_u = imgs.to(device), masks.to(device), imgs_u.to(device)

            with torch.amp.autocast(device_type='cuda'):
                s_logits = student(imgs)

                # weighted BCEWithLogits
                weight_map = masks*pos_weight + (1-masks)*neg_weight
                loss_sup = F.binary_cross_entropy_with_logits(s_logits, masks, weight=weight_map)

                with torch.no_grad(): t_logits=teacher(imgs_u)
                u_logits=student(imgs_u)
                loss_cons=F.mse_loss(torch.sigmoid(u_logits),torch.sigmoid(t_logits))

                loss=(loss_sup+cons_w*loss_cons)/accumulate_steps
            
            scaler.scale(loss).backward()

            if (count+1)%accumulate_steps==0:
                scaler.step(optimizer); scaler.update()
                for tp,sp in zip(teacher.parameters(),student.parameters()): tp.data.mul_(ema_decay).add_(sp.data,alpha=1-ema_decay)
                optimizer.zero_grad()

            tot_loss+=loss.item()*accumulate_steps; count+=1
            loop.set_postfix(loss=(tot_loss/count))


        avg_loss=tot_loss/count
        print(f"Epoch {ep}/{epochs} Avg Loss: {avg_loss:.4f}")

        if avg_loss<best_loss:
            best_loss=avg_loss; best_state=student.state_dict(); no_imp=0
        else:
            no_imp+=1
            if no_imp>=patience:
                print(f"Stop: no improvement in {patience} epochs")
                break
        scheduler.step(avg_loss)

    torch.save(best_state, save_name)
    print(f"Done. Best Loss={best_loss:.4f}, model saved to {save_name}")

if __name__=="__main__":
    main()
