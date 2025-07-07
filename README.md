# SemiSeg
This repo contains codes to train a semi-supervised models to binary segment cells from brightfield images. It uses Multi-Scale Unet to perform the segmentation. It also contains files to match image annotations from Label Studio to image names and preprocess the brightfield images. Post-processing files can summarize the largest sizes of organoids in each image. See file descriptions for more.

## Project Contributors <br>
**Dr. Adam Farsheed** (Salk Institute): Initiated the project, provided the brightfield images of organoids, and offered domain knowledge for the project. <br>

**Yang Han** (UCSD): Designed and trained the model, wrote all codes, annotated brightfield images for training the model, and created Github repo for the project. <br>

**If you find this package useful, please cite us!** <br>
Citation: Farsheed, A., Han, Y., *SemiSeg*, Github repository, https://github.com/TonyYangHan/SemiSeg

If you encounter issues when using the code, please contact Yang Han (*yah015@ucsd.edu*)

## Content overview
1. `Semi_supervised_v2.py` contains all source codes to train a semi-supervised machine learning model with a small quantity of image-mask pairs and majority of unlabeled raw brightfield images.

2. `Detection.ipynb` allow your to load your trained model and a directory of unprocessed brightfield images to detect organoids in the image and measure their sizes (in pixels).

3. `Test.ipynb` allow you to load a directory of brightfield images and a directory of corresponding masks to assess the raw performance of the detection model.

4. `Preprocess.ipynb` allows you matching annotations from Label Studio to their corresponding images and change batches of file names easily.

5. `best_student_v2.pth` saves the best trained model for detection of organoids in brightfield images (ready to be loaded and used) Downloadable [here](https://drive.google.com/file/d/1EdUFLkr4VUbRDltGK2Inpj6Hw1HRxpgX/view?usp=sharing)

## Hardware recommendations
This package is somewhat computationally demanding. Therefore, running codes with CUDA GPU (most modern NVIDIA GPUs) is **strongly recommended**. It can speed up the run time very significantly, though running on CPU is entirely possible <br>

Cloud computing platforms, such as Google Colab or your own HPC access, are great choices if you don't have CUDA GPU on your own device.

## Step to use this repo
1. Set up the conda environment for running the code files (if haven't already)<br>
Install miniconda if you have not already. It is a great tool for you to manage the package dependencies so you don't break other package when you change some packages. It is free and easy to install + use.
You may find the installation instruction [here](https://www.anaconda.com/docs/getting-started/miniconda/install#linux).

For Windows user, it is recommended to install WSL2 and install miniconda on there. Mac users, please use your terminal (you got it for free). If your system is entirely Linux (e.g. Cloud computers), just install it. 

ChatGPT can be your great friend in guiding through the installation process. <br>

Once you see this it means installation is successful <br>
```
(base) your_username@your_device:<current_directory>$
```

<br>

2. Download this repo as .zip and decompress it to your desired directory. Change the working directory in the terminal to the desired directory.

Use this command in the terminal to conveniently install all packages
Just copy paste into terminal and hit enter. Make sure `computer_vision.yaml` is under your current working directory (simply run `pwd` to know where you are at or change directory as appropriate) <br>
<br>
```
conda env create -f computer_vision.yaml
```

Use this to activate the environment
```
conda activate computer_vision
```
Now you should see
```
(computer_vision) your_username@your_device:<your_directory>$
```

It is recommended to put `best_student_v2.pth` in the same folder as this package.

3. Launch Jupyter notebook session
Simply type and run <br>
`jupyter notebook` <br>
and click on the link printed in the terminal, you will enter a graphical interface where you can just open and run `.ipynb` files

4. Run detection on images
Once you open `Detect.ipynb`, you can follow the comments to set all parameters to yours and go to `Run` on the top of the bar and click on `Run All Cells`. 

## (Optional) Train your own model with your own data.
You can train your own model using your own brightfield images and annotations. The model would better suit your needs if training is done correctly.

1. If you have limited knowledge with semi-supervised learning, I suggest you to learn a little about it as it will be beneficial for understanding the training process.

2. To train an AI model, you need to provide raw images and their corresponding masks (binary annotations) as their "study materials." Fortunately, training a useful semi-supervised detect model only requires you to annotate ~10% of all images, in general. Make sure to select out train images with high-quality and diverse lighting/contrast conditions. The first step is to label images using Label Studio (free software with instructions on YouTube). Remember to download the annotation `.json` file and export all annotated masks as `.png`.

3. Set the `labeled_imgs` variable to the directory of brightfield image you selected to annotate. (Use `Preprocess.ipynb` to remove the space with underscores) Set `mask_imgs` to the directory of corresponding masks you generated. (Use `Preprocess.ipynb` to match the names of annotations to the images) Set `unlabeled_imgs` to directory of the unlabeled brightfield images.

4. Tweak parameters as needed. After that, use this to start training the model
```
python Semi_supervised_v2.py
```

5. Watch how average loss change over epochs. Tweak the model as needed. ChatGPT or other GenAI can offer you advice on how to tune the model as long as you described the result in great detail with them. 



