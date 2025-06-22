# Removing Adversarial Noise in  Class Activation <br> Feature Space (ICCV 2021)


深度神经网络（DNNs）易受对抗性噪声的影响。基于预处理的防御方法通常通过对输入数据进行处理来消除对抗性噪声，但这类方法普遍存在误差放大效应的问题，尤其在面对持续演变的攻击时表现更为明显。为解决这一问题，本文提出了一种在类别激活特征空间中实现自监督对抗训练的机制来消除对抗性噪声。具体而言，我们首先通过最大化对自然样本类别激活特征的干扰来生成对抗样本，随后训练一个去噪模型，以最小化对抗样本与自然样本在类别激活特征空间中的距离。



##  Environment

The code was tested on a Conda environment installed on Ubuntu 18.04.

Install [Conda](https://docs.conda.io/en/latest/miniconda.html) and then create an environment as follows:

* Python 3.6.8
* numpy 1.19.5
* pandas 1.1.5
* Pillow 9.0.0
* scipy 1.5.4
* statsmodels 0.12.2
* torch 1.8.0+cu111
* torchvision 0.9.0+cu111

## Datasets

- Please download and place all datasets into the 'data' directory. 

- Download the [`cifar10`](<https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>) 

- Download the [`cifar100`](<https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz>)

  ```
    cifar-10-batches-py/
    ├── batches.meta      # 包含类别名称（'airplane', 共10类）
    ├── data_batch_1      # 训练集批次1（10,000样本）
    ├── data_batch_2      # 训练集批次2
    ├── data_batch_3      # 训练集批次3
    ├── data_batch_4      # 训练集批次4
    ├── data_batch_5      # 训练集批次5
    └── test_batch        # 测试集（10,000样本）

    cifar-100-python/
    ├── meta              # 细粒度类别（100类）和粗粒度类别（20类）
    ├── train             # 训练集（50,000样本）
    └── test              # 测试集（10,000样本）
  ```

## Preparation

```
Python train_target_model.py 
```

This code provides three model architectures (including VggNet, ResNet and Wide-ResNet). The trained model will be saved in the "checkpoint" folder.


- To generate adversarial training data

-For Training data

```
python example_cam.py
```

We use the "Class Activation Feature based Attack" (CAFD) to generate adversaial samples. The generated samples will be saved in the 'data/training' folder.

-For Test data

```
python example_other.py or python example_autoattack.py
```

We use the "[advertorch](https://github.com/BorealisAI/advertorch)" toolbox to help generate adversairal samples. The first code provides ![](http://latex.codecogs.com/svg.latex?L_{\infty}) PGD, ![](http://latex.codecogs.com/svg.latex?L_{2}) CW, [DDN](https://arxiv.org/abs/1811.09600), [STA](https://openreview.net/forum?id=HyydRMZC-), etc., to generate different adversarial samples. The second code provides [Autoattack](https://arxiv.org/abs/2003.01690).
The generated samples will be saved in the "data/test" folder.

## Train
To train the CAFD

```
python train_or_test_denoiser.py --mode 0
```

The model parameters of the used target model comes from "checkpoint" folder. The trained defense model will be saved in "checkpoint_denoise" folder.


## Test

- To test CAFD

```
python train_or_test_denoiser.py --mode 1
```

The input data comes from "data/test" folder, and the denoised data is saved in "results/defense" folder. 

- To compute the accuracy rate

```
python test.py
```

## Acknowledgement

We would like to thank the authors of [CFAD]( https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_Removing_Adversarial_Noise_in_Class_Activation_Feature_Space_ICCV_2021_paper.pdf) ,providing code for academic practice

