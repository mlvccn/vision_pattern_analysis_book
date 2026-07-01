# Squeeze Training for Adversarial Robustness (ICLR 2023)



对抗样本在训练过程中可以以多种方式被利用，从而衍生出各种对抗训练方法。本文考虑到对抗样本和协作样本都是由损失函数景观（Loss Landscape）的非平坦性所引起的，因此提出：对每个原始样本（Benign Example）的 $\epsilon$邻域内任意两个数据点之间可能产生的最大输出差异进行惩罚。受诸如 TRADES 中提出的对抗正则化思想的启发，本文采用了一个针对原始样本预测损失的表达式，并结合一个对潜在的对抗样本和协作样本联合进行正则化的表达式。由于在该正则化机制中，并不特别要求良性样本本身参与正则化项中的误差累积，因此良性样本的输出既不会显式地被鼓励变得“对抗”，也不会被鼓励变得“协作”。



### Installation

The code was tested on a Conda environment installed on Ubuntu 18.04.

Install [Conda](https://docs.conda.io/en/latest/miniconda.html) and then create an environment as follows:

## Environments
* Python 3.6.8
* numpy 1.19.5
* pandas 1.1.5
* Pillow 9.0.0
* scipy 1.5.4
* statsmodels 0.12.2
* torch 1.8.0+cu111
* torchvision 0.9.0+cu111

## Datasets

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


## Train
Train a ResNet-18 on CIFAR-10/CIFAR-100/SVHN using our ST:
```
dataset=${dataset} beta=${beta} epochs=${epochs} lossmode=${lossmode} bash st.sh
```
where ```${dataset}``` in ```["cifar10", "cifar100", "svhn"]```, ```${epochs}``` in ```["120", "80"]```, and ```${lossmode}``` in ```["js","l2","symmkl"]```.

Train a WRN-28-10 on CIFAR-10 using ST-RST:
```
bash st_rst.sh
```

## Test
***FGSM, PGD and CW for ST trained ResNet-18***:
```
python3 test.py --model-path ${modelpath} --log ${logpath} --dataset ${dataset}
```
***PGD<sub>RST</sub> for ST-RST trained WRN-28-10***:
```  
python3 test.py --model_path ${modelpath} --attack pgd --output_suffix pgd_rst
```
***AutoAttack***
```  
python3 test_aa.py --model-path ${modelpath} --arch ${architecture} --dataset ${dataset} --log ${logpath}
```
where ```${architecture}``` in ```["resnet18", "wrn-28-10"]```.

## Acknowledgement

We would like to thank the authors of [TT]( https://github.com/qizhangli/ST-AT) ,providing code for academic practice

