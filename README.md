# Fully Convolutional Siamese Networks for Change Detection

---
This repo has been deprecated. Please see [CDLab](https://github.com/Bobholamovic/CDLab), which includes more architectures and datasets.
---

This is an unofficial implementation of the paper

> Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch. (2018, October). Fully convolutional siamese networks for change detection. In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.
 
~~as the [official repo](https://github.com/rcdaudt/fully_convolutional_change_detection) does not provide the training code.~~

[paper link](https://ieeexplore.ieee.org/abstract/document/8451652)

# Dependencies

> opencv-python==4.1.1  
  pytorch==1.3.1  
  torchvision==0.4.2  
  pyyaml==5.1.2  
  scikit-image==0.15.0  
  scikit-learn==0.21.3  
  scipy==1.3.1  
  tqdm==4.35.0

Tested using Python 3.7.4 on Ubuntu 16.04 and Python 3.6.8 on Windows 10.

# Basic usage

```bash
# The network definition scripts are from the original repo
git clone --recurse-submodules git@github.com:Bobholamovic/FCN-CD-PyTorch.git
cd FCN-CD-PyTorch
mkdir exp
cd src
```

In `src/constants.py`, change the dataset locations to your own. In `config_base.yaml`, set specific configurations.

For training, try

```bash
python train.py train --exp_config ../configs/config_base.yaml
```

For evaluation, try

```bash
python train.py eval --exp_config ../configs/config_base.yaml --resume path_to_checkpoint --save-on
```

You can check the model weight files in `exp/base/weights/`, the log files in `exp/base/logs`, and the output change maps in `exp/base/out`.

---
# Changed

- 2020.3.14 Add configuration files.
- 2020.4.14 Detail README.md.
- 2020.12.8 Update framework.
