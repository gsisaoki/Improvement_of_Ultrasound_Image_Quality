<img src="figure/thumbnail.jpg">

# Image Quality Improvement in Single Plane-Wave Imaging Using Deep Learning

## Overview
&nbsp;&nbsp;&nbsp;In ultrasound image diagnosis, single plane-wave imaging (SPWI), which can acquire ultrasound images at more than 1,000 fps, has been used to observe detailed tissue and evaluate blood flow.
SPWI achieves high temporal resolution by sacrificing the spatial resolution and contrast of ultrasound images.
To improve spatial resolution and contrast in SPWI, coherent plane-wave compounding (CPWC) is used to obtain high-quality ultrasound images, i.e., compound images, by adding radio frequency (RF) signals acquired by transmitting plane waves in different directions.
Although CPWC produces high-quality ultrasound images, their temporal resolution is lower than that of SPWI.
To address this problem, some methods have been proposed to reconstruct an image comparable to a compund image from RF signals obtained by transmitting a small number of plane waves in different directions.
These methods do not fully consider the characteristics of RF signals, resulting in lower image quality compared to a compound image.
In this paper, we propose a method to reconstruct high-quality ultrasound images in SPWI by considering the characteristics of RF signals of a single plane wave to obtain ultrasound images of the equivalent quality as CPWC.
The proposed method employs an encoder-decoder model that combines 1D U-Net and 2D U-Net to consider that the point spread functions of RF signals depend on the depth.
High-quality ultrasound images can be reconstructed from RF signals in SPWI by training the encoder-decoder model to minimize the loss that considers the point spread functions and frequency characteristics of RF signals.
Through a set of experiments using the public dataset and our dataset, we demonstrate that the proposed method can reconstruct higher-quality ultrasound images from RF signals in SPWI than conventional methods.

## Installation  
You can get the source code in either of the following two ways.
- Run the following command.
```
git clone https://github.com/gsisaoki/US_Pose_Estimation
```
- Click [this URL](https://github.com/gsisaoki/US_Pose_Estimation/archive/refs/heads/main.zip) and unzip the download file.

## Preparation

```
cd US_Pose_Estimation
mkdir src
mv * src/
cd src
```

## Requirements  
  * Python 3.9
  * pytorch 1.10

Run the following commands to easily train or test the probe pose estimation methods,
```
conda create -n uspose python=3.9
conda activate uspose
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Dataset

- Overview  
A total of 2,880 US sequences (720,000 frames) are acquired under 48 acquisition conditions from two US phantoms (a breast phantom and a hypogastric phantom) as shown in the table below.

- Download  
In order to train or test the probe pose estimation methods using USPose, you need to download USPose from [our project page]().  
※ Warning: It weights about 500GB, make sure you have enough space to unzip too.  

## Usage
1. Rewrite root_path (absolute path to src, line 79) and dataset_path (absolute path to downloaded USPose, line 80) in [main.py](./main.py).
2. If you want to train CNN using GPU under same conditions as in our paper , run the following command. You can run the code according to the 7 evaluation protocols described in our paper. See [our paper]() for details. The available 'model_name' are ['resnet18'](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf), ['prevost'](https://link.springer.com/content/pdf/10.1007/978-3-319-66185-8_71.pdf), ['miura2020'](https://link.springer.com/content/pdf/10.1007/978-3-030-60334-2_10.pdf), ['guo'](https://link.springer.com/content/pdf/10.1007/978-3-030-59716-0_44.pdf), and ['miura2021'](https://link.springer.com/content/pdf/10.1007/978-3-030-87583-1_10.pdf). When using 'miura2020' or 'miura2021', download the traind models (**flownets_EPE1.951.pth.tar**) from [FlowNet repository](https://github.com/ClementPinard/FlowNetPytorch) to [models/](./models/). In order to change the experimental conditions, rewrite the respective 'train.yaml' file in [yaml](./yaml).  

    - Protocol 1  
    `sh exp/prot1.sh train model_name`
    - Protocol 2  
    `sh exp/prot2.sh train model_name`
    - Protocol 3  
    `sh exp/prot3.sh train model_name`
    - Protocol 4  
    `sh exp/prot4.sh train model_name`
    - Protocol 5  
    `sh exp/prot5.sh train model_name`
    - Protocol 6  
    `sh exp/prot6.sh train model_name`
    - Protocol 7  
    `sh exp/prot7.sh train model_name`

3. If you want to test **our trained models**, you need to download the respective trained models from [here](). Set the 'checkpoint_path' in 'test.yaml' file in [yaml](./yaml) to the path of the downloaed model and run the following command.

    - Protocol 1  
    `sh exp/prot1.sh test model_name`
    - Protocol 2  
    `sh exp/prot2.sh test model_name`
    - Protocol 3  
    `sh exp/prot3.sh test model_name`
    - Protocol 4  
    `sh exp/prot4.sh test model_name`
    - Protocol 5  
    `sh exp/prot5.sh test model_name`
    - Protocol 6  
    `sh exp/prot6.sh test model_name`
    - Protocol 7  
    `sh exp/prot7.sh test model_name`

4. If you want to test **your own trained models**, set the 'checkpoint_path' in 'test.yaml' file in [yaml](./yaml) to the path of the saved model in 'USPose/result/' directory and run the above command.

## Results
In order to visualize 3D US images, you can use the matlab code [make_3dus.m](./lib/make_3dus.m). Set the 'subject_name', 'result_path', and 'dataset_path' in [make_3dus.m](./lib/make_3dus.m).
After running [make_3dus.m](./lib/make_3dus.m), two files ('3d_volume.sxi' and '3d_volume.sw') are generated.
You open '~.sxi' file to visualize the reconstructed 3D US image using [Stradview](https://mi.eng.cam.ac.uk/Main/StradView).
Stradview is a free medical 3D data visualisation package which can run under Windows or Linux.


## References
- R. Prevost, M. Salehi, J. Sprung, A. Ladikos, R. Bauer, and W. Wein, “Deep learning for sensorless 3D freehand ultrasound imaging,” Proc. Int’l Conf. Medical Image Computing and Computer-Assisted Interven- tion, vol. 10434, pp. 628–636, Sep. 2017.

- K. Miura, K. Ito, T. Aoki, J. Ohmiya, and S. Kondo, “Localizing 2D ultrasound probe from ultrasound image sequences using deep learning for volume reconstruction,” Proc. Int’l Workshop on Advances in Simplifying Medical Ultrasound, vol. 12437, pp. 97–105, Oct. 2020.

- H. Guo, S. Xu, B. Wood, and P. Yan, “Sensorless freehand 3D ultrasound reconstruction via deep contextual learning,” Proc. Int’l Conf. Medical Image Computing and Computer-Assisted Intervention, vol. 12263, pp. 463–472, Oct. 2020.

- K. Miura, K. Ito, T. Aoki, J. Ohmiya, and S. Kondo, “Pose estimation of 2D ultrasound probe from ultrasound imagesequences using CNN and RNN,” Proc. Int’l Workshop on Advances in Simplifying Medical Ultrasound, vol. 12967, pp. 96–105, Sep. 2021.

## Acknowledgments
Parts of this code were deribed, as noted in the code, from [ClementPinard/FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch), [ndrplz/ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch), and [DIAL-RPI/FreehandUSRecon](https://github.com/DIAL-RPI/FreehandUSRecon).
