# Few Sample Knowledge Distillation for Efficient Network Compression

This repository contains the samples code for FSKD, [Few Sample Knowledge Distillation for Efficient Network Compression](https://arxiv.org/abs/1812.01839) (CVPR 2020) by Tianhong Li, Jianguo Li, Zhuang Liu and Changshui Zhang.

The repo shows how to train a VGG-16 model on CIFAR-10 and then prune it with very few unlabeled samples using FSKD. It can also be extended to other models, network pruning / decoupling methods and datasets.

## Installation

The code was tested on a Conda environment installed on Ubuntu 18.04.

Install [Conda](https://docs.conda.io/en/latest/miniconda.html) and then create an environment as follows:

- Python 3.8.19 :

```
conda create -n slpkd python=3.8.19 
```

```
conda activate slpkd
```

- Pytorch 1.7
- CUDA 10.1

```
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
```

### CIFAR-10

- Follow the instructions [here](https://www.cs.toronto.edu/~kriz/cifar.html) to download the dataset.

  ```
  KD/
  └── Cifar10/
    │   ├── cifar-10-python
    │   │   └──train
    │   │   └──test
  ```

## Training VGG-16

```shell
python main.py --dataset cifar10 --arch vgg --depth 16 --lr 0.01
```

## Prune and FSKD

```shell
python vggprune_pruning.py --dataset cifar10 --depth 16 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT] --num_sample 500
```

## main.py

This file performs training of different network structure. You can specify the training parameter as well as whether using sparsity training or not.

## vggprune_pruning.py

This file performs FSKD on specified VGG pretrained model. You can specify the number of samples used here. The script first build up the pruned VGG model from the original model. It then add the additional 1x1 conv layer, performs FSKD layer by layer (recover_one_layer()), and finally absorb the additional 1x1 conv layer back to the model.

## Acknowledgement

We would like to thank the authors of [FSKD](Few Sample Knowledge Distillation for Efficient Network Compression) which has significantly accelerated the development of our book.