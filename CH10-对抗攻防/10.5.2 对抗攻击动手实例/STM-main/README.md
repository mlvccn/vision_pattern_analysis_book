# Improving the Transferability of Adversarial Examples with Arbitrary Style Transfer (ACM MM 2023)

深度神经网络容易受到通过在干净输入上施加人眼难以察觉的扰动而生成的对抗样本的影响。尽管许多攻击方法在白盒环境下能实现很高的攻击成功率，但这些方法在黑盒环境下的迁移性往往较弱。最近，研究者提出了多种提升对抗迁移性的方法，其中输入变换是最有效的技术路线之一。在本研究中，我们发现现有基于输入变换的方法主要采用同领域数据增强。受领域泛化研究的启发，我们尝试通过跨领域数据增强来进一步提升迁移性。具体而言，风格迁移网络可以在保持图像语义内容不变的情况下改变其底层视觉特征的分布。因此，我们提出了一种名为风格迁移攻击法（Style Transfer Method，STM）的新型攻击方法，该方法利用我们设计的任意风格迁移网络将图像转换到不同域。为避免风格化图像对分类网络产生语义不一致的问题，我们对风格迁移网络进行微调，并将添加了随机噪声的生成图像与原始图像混合，以保持语义一致性并增强输入多样性。

### Installation

The code was tested on a Conda environment installed on Ubuntu 18.04.

Install [Conda](https://docs.conda.io/en/latest/miniconda.html) and then create an environment as follows:

## Environments
* python == 3.7.11
* pytorch == 1.8.0
* torchvision == 0.8.0
* numpy == 1.21.2
* pandas == 1.3.5
* opencv-python == 4.5.4.60
* scipy == 1.7.3
* pillow == 8.4.0
* pretrainedmodels == 0.7.4
* tqdm == 4.62.3
* imageio == 2.6.1

## Datasets

- Download the [`datasets`](<https://github.com/Zhijin-Ge/STM/tree/main/dataset>)  

- Put the data in **'./dataset/'**.

## Attack

```
python Incv3_STM_Attacks.py
```

## Test

```
python verify.py
```

## Acknowledgement

We would like to thank the authors of [STM]( https://github.com/Zhijin-Ge/STM?tab=readme-ov-file) ,providing code for academic practice

