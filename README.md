<img src="figure/thumbnail.jpg">

# Image Quality Improvement in Single Plane-Wave Imaging Using Deep Learning

## Introduction
&nbsp;&nbsp;&nbsp;We propose a method to improve the image quality of ultrasound image acquired with a single plane wave in single plane-wave imaging.
The proposed method employs the encoder-decoder model combining 1D U-Net and 2D U-Net [^1] to consider that the point spread function of RF signals in the lateral direction varies with depth.
The encoder-decoder model is trained using the frequency loss considering amplitude and phase obtained by 1D discrete Fourier transform of the RF signal in the axial direction.
For more details, please refer to [our paper](https://github.com/gsisaoki/Improvement_of_Ultrasound_Image_Quality).

This repository contains the following used for the results in our paper:
- PyTorch implementation of the proposed method
- Our original label of wire and gray-scale targets in ultrasound images
- Trained models of the proposed method

## Requirements  
  * Python 3.9
  * pytorch 1.10

Run the following commands to easily train or test the proposed method,
```
conda env create -n {env_name} -f env.yml
conda activate {env_name}
```

## Preparation
You can get the source code in either of the following two ways.
- Run the following command.
```
git clone https://github.com/gsisaoki/Improvement_of_Ultrasound_Image_Quality.git
```
- Click [this URL](https://github.com/gsisaoki/Improvement_of_Ultrasound_Image_Quality/archive/refs/heads/master.zip) and unzip the download file.

After you download the source code, run the following command.
```
cd Improvement_of_Ultrasound_Image_Quality-master
mkdir src
mv * src/
cd src
```

## Dataset

In order to download our dataset, please send an e-mail to <a href="mailto:kanta@aoki.ecei.tohoku.ac.jp">kanta@aoki.ecei.tohoku.ac.jp</a>.<br>
When we receive the e-mail, we will share the download link.
Our original labels obtained by annotation are included in [coord.json](./coord.json).
You can use "Evaluate" class included in [utils.py](./utils.py) to evaluate the quality of ultrasound images.

## Dataset layout

After unzipping the downloaded zip file, you will get directory with the following structure.
```
 Dataset.zip
 ├── BP                          # Directory of the breast ultrasound phantom
 |   └── {seq_id}_seq
 |        └── {frame_id}_frame
 |             ├── rf_real
 |             |   ├── input.mat # .mat file which contains the real part of the single-plane wave RF data
 |             |   └── comp.mat  # .mat file which contains the real part of the compounded RF data
 |             └── rf_imag
 |                 ├── input.mat # .mat file which contains the imaginary part of the single-plane wave RF data
 |                 └── comp.mat  # .mat file which contains the imaginary part of the compounded RF data
 ├── QAP                         # Directory of the ultrasound image assurance phantom
 |
 └── Invivo                      # Directory of the healthy subjects
```

## Trained models
The trained model of each method is available from the link below.

- Li et al.'s method[^2]: [Google Drive](https://drive.google.com/file/d/1cM3SAETD2We14dlFojw6eAHMmlgQQm-V/view?usp=sharing) (153MB)

- 1D U-Net: [Google Drive](https://drive.google.com/file/d/14ThMXJwG1tyJHrZcN1hpOcVu58LXjyv6/view?usp=sharing) (96MB)

- 2D U-Net: [Google Drive](https://drive.google.com/file/d/1pg47XUWWA1iEgsa1GzORKJQYDfn0K0cc/view?usp=sharing) (278MB)

- Proposed method: [Google Drive](https://drive.google.com/file/d/1i5k8pl41aun3r3ihA79fIxEJeKdkiPLp/view?usp=sharing) (375MB)

## Usage
1. Rewrite root_path (absolute path to src, line 83) and dataset_path (absolute path to downloaded dataset, line 80) in [main.py](./main.py).
2. If you want to train CNN using GPU under same conditions as in our paper , run the following command. In order to change the experimental conditions, rewrite the respective 'train.yaml' file in [yaml](./yaml).  

    `sh exp/train.sh`

3. If you want to test **our trained models**, you need to download the trained model. Set the 'checkpoint_path' in 'test.yaml' file in [yaml](./yaml) to the path of the downloaed model and run the following command.

    `sh exp/eval.sh`

4. If you want to test **your own trained models**, set the 'checkpoint_path' in 'test.yaml' file in [yaml](./yaml) to the path of the saved model in 'USPose/result/' directory and run the above command.

## References
[^1]: O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," Proc. Int'l Conf. Medical Image Computing and Computer Assisted Intervention, Springer, LNCS, vol. 9351, pp. 234--241, Oct. 2015.
[^2]: Z. Li, A. Wiacek, and M. Bell, “Beamforming with deep learning from single plane wave RF data,” Proc. IEEE Int’l. Ultrasonics Symposium, pp. 1-4, Sep. 2020.