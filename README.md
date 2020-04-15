# Fully Convolutional Siamese Networks for Change Detection

This is an unofficial implementation of the paper

> Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch. (2018, October). Fully convolutional siamese networks for change detection. In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.
 
as the [official repo](https://github.com/rcdaudt/fully_convolutional_change_detection) does not provide the training code. 

[paper link](https://ieeexplore.ieee.org/abstract/document/8451652)

# Prerequisites

> opencv-python==4.1.1  
  pytorch==1.2.0  
  pyyaml==5.1.2  
  scikit-image==0.15.0  
  scikit-learn==0.21.3  
  scipy==1.3.1  
  tqdm==4.35.0  

Tested on Python 3.7.4, Ubuntu 16.04 and Python 3.6.8, Windows 10.

# Basic usage

```bash
# The network definition scripts are from the original repo
git clone --recurse-submodules git@github.com:Bobholamovic/FCN-CD-PyTorch.git
cd FCN-CD-PyTorch
mkdir exp
cd src
```

In `src/constants.py`, change the dataset directories to your own. In `config_base.yaml`, feel free to change the configurations.

For training, try

```bash
python train.py train --exp-config ../config_base.yaml
```

For evaluation, try

```bash
python train.py val --exp-config ../config_base.yaml --resume path_to_checkpoint --save-on
```

You can find the checkpoints in `exp/base/weights/`, the log files in `exp/base/logs`, and the output change maps in `exp/base/outs`.

# Train on Air Change dataset and OSCD dataset

To carry out a full training on these two datasets and with all three architectures, run the `train9.sh` script under the root folder of this repo.
```bash
. ./train9.sh
```

And check the results in different subdirectories of `./exp/`. 

# Create your own configuration file

During scientific research, it is common case that we have to do a lot of experiments with different settings, and that's why we need the configuration files to better manage those settings. In this repo, you can create a `yaml` file under the naming convention below:

`config_TAG{_SUFFIX}.yaml`

Those in the curly braces can be omitted. `TAG` usually stands for an experiment group. For example, a set of experiments for an architecture, a dataset, etc. It will be the name of the subdirectory that holds all the checkpoints, log files, and output images. `SUFFIX` can be used to distinguish different experiments in an experiment group. If it is specified, the generated files of this experiment will be tagged with `SUFFIX` in their file names. In plain English, `TAG1` and `TAG2` have major differences, while `SUFFIX1` and `SUFFIX2` of the same `TAG` share most of the configurations. By combining `TAG` and `SUFFIX`, it is convenient for both coarse-grained and find-grained control of experimental configurations.

Here is an example to help you understand. Suppose I'm going to finish my experiments on two datasets, OSCD and Lebedev, and I'm not sure which batch size achieves best performance. So I create these 5 config files.
```
config_OSCD_bs4.yaml
config_OSCD_bs8.yaml
config_OSCD_bs16.yaml
config_Lebedev_bs16.yaml
config_Lebedev_bs32.yaml
```

After training, I get my `exp/` folder like this:

```
-exp/
--OSCD/
---weights/
----model_best_bs4.pth
----model_best_bs8.pth
----model_best_bs16.pth
---outs/
---logs/
---config_OSCD_bs4.yaml
---config_OSCD_bs8.yaml
---config_OSCD_bs16.yaml
--Lebedev/
---weights/
----model_best_bs16.pth
----model_best_bs32.pth
---outs/
---logs/
---config_Lebedev_bs16.yaml
---config_Lebedev_bs32.yaml
```

Now the experiment results are organized in a more structured way, and I think it would be a little bit easier to collect the statistics. Also, since the historical experiments are arranged in neat order, you will soon remember what you'd done when you come back to these results, even after a long time.

Alternatively, you can configure from the command line. This can be useful when there is only minor change between two single runs, because the configuration items from the command line is set to overwrite those from the `yaml` file. That is, the final value of each configuration item is evaluated and applied in the following order:

```
default_value -> value_from_config_file -> value_from_command_line
```

At least one of the above three values should be given. In this way, you don't have to include all of the config items in the `yaml` file or in the command-line input. You can use either of them, or combine them. Make your choice according to preference and circumstances.

---
# Changed

- 2020.3.14 Add the configuration files of my experiments. 
- 2020.4.14 Detail README.md.
