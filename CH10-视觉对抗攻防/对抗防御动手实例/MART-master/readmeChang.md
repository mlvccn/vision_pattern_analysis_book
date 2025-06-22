# Improving Adversarial Robustness Requires Revisiting Misclassified Examples (ICLR 2020)



MART算法由Wang等人提出，是对抗训练领域针对模型鲁棒性优化的代表性方法之一。该方法通过显式区分对抗样本中的易误分类样本（misclassified examples）与难样本（hard examples），动态调整训练损失函数的权重分配，从而提升模型对对抗攻击的防御能力。具体而言，模型通过最小化正常样本的交叉熵损失与对抗样本的误分类感知损失（Misclassification-Aware Loss）的加权组合，实现对对抗扰动下决策边界的最优修正。此方法侧重于对抗训练中样本的误分类敏感性差异，适用于对抗样本分布复杂且攻击强度动态变化的场景。



### Installation

The code was tested on a Conda environment installed on Ubuntu 18.04.

Install [Conda](https://docs.conda.io/en/latest/miniconda.html) and then create an environment as follows:

- Python 3.8.19 :

```
conda create -n mart python=3.8.19 
```

```
conda activate mart
```

- Pytorch 1.7
- CUDA 10.1

```
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
```


```setup
pip install opencv-python
```

## CNNs

### CIFAR-10/CIFAR-100

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


1. For ResNet18

  ```
python3 train_resnet.py for ResNet18
  ```

2. For WideResNet

  ```
python3 train_wideresnet.py for WideResNet
  ```


## Acknowledgement

We would like to thank the authors of [MART]( https://github.com/YisenWang/MART) ,providing code for academic practice