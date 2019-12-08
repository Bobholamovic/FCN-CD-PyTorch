# Fully Convolutional Siamese Networks for Change Detection

This is an unofficial implementation of the paper

> Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch. (2018, October). Fully convolutional siamese networks for change detection. In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.
 
as the [official repo](https://github.com/rcdaudt/fully_convolutional_change_detection) does not provide the training code. 

[paper link](https://ieeexplore.ieee.org/abstract/document/8451652)

# Usage

```bash
# The network definition scripts are from the original repo
git clone --recurse-submodules git@github.com:Bobholamovic/FCN-CD-PyTorch.git   
```

```bash
cd src
python train.py train --exp-config ../config_base.yaml
```