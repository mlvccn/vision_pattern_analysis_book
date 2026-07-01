# Distilling the knowledge in a neural network&&FitNets: Hints for thin deep nets



最早由Hinton等人提出的Logits蒸馏是知识蒸馏领域的基础方法之一，该方法通过对教师模型输出的softmax概率分布（即logits经过温度调节后的软标签）进行拟合，使学生模型学习到教师模型对各类别间相似度的细致表达。具体而言，学生模型通过最小化其预测概率分布与教师模型预测概率分布之间的Kullback-Leibler散度损失，实现对教师知识的有效迁移。此方法侧重于模型最终决策层的知识传递，适用于分类任务中类别概率信息丰富且稳定的场景。

中间特征蒸馏则进一步拓展了知识传递的层次，旨在通过约束学生模型在中间层提取的特征表示与教师模型的对应层特征保持高度一致，从而强化学生模型的特征表达能力。此类方法通常包括特征响应匹配、特征关系矩阵对齐及基于特征图的辅助损失函数设计，能够捕捉教师模型在中间层隐含的结构化信息与语义特征。中间特征蒸馏适用于教师与学生模型结构相似或者希望学生在表达能力上更接近教师的应用场景，且被广泛应用于图像识别、目标检测等任务中。



### Installation

The code was tested on a Conda environment installed on Ubuntu 18.04.

Install [Conda](https://docs.conda.io/en/latest/miniconda.html) and then create an environment as follows:

- Python 3.8.19 :

```
conda create -n slpkd python=3.8.19 
```

```
conda activate kd
```

- Pytorch 1.7
- CUDA 10.1

```
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
```


```setup
pip install opencv-python
```

## Distilling CNNs

### CIFAR-100

- Download the [`cifar_teachers.tar`](<https://github.com/megvii-research/mdistiller/releases/tag/checkpoints>) and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

- Follow the instructions [here](http://www.cs.toronto.edu/~kriz/cifar.html) to download the dataset.

  ```
  KD/
  └── Cifar100/
    │   ├── cifar-100-python
    │   │   └──train
    │   │   └──test
  ```


1. For KD

  ```bash
  # KD
  python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml
  # KD+Ours
  python tools/train.py --cfg configs/cifar100/kd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
  ```

2. For DKD

  ```bash
  # DKD
  python tools/train.py --cfg configs/cifar100/dkd/resnet32x4_resnet8x4.yaml 
  # DKD+Ours
  python tools/train.py --cfg configs/cifar100/dkd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
  ```
3. For MLKD

  ```bash
  # MLKD
  python tools/train.py --cfg configs/cifar100/mlkd/resnet32x4_resnet8x4.yaml
  # MLKD+Ours
  python tools/train.py --cfg configs/cifar100/mlkd/resnet32x4_resnet8x4.yaml --logit-stand --base-temp 2 --kd-weight 9 
  ```

4. For CTKD

Please refer to [CTKD](./CTKD).

#### Results and Logs

We put the training logs in `./logs` and hyper-linked below. The name of each log file is formated with `KD_TYPE,TEACHER,STUDENT,BASE_TEMPERATURE,KD_WEIGHT.txt`. The possible third value for DKD is the value of BETA. Due to average operation and randomness, there may be slight differences between the reported results and the logged results. 

1. Teacher and student have **identical** structures:

| Teacher <br> Student |ResNet32x4 <br> ResNet8x4|VGG13 <br> VGG8|Wrn_40_2 <br> Wrn_40_1|Wrn_40_2 <br> Wrn_16_2|ResNet56 <br> ResNet20|ResNet110 <br> ResNet32|ResNet110 <br> ResNet20|
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:------------------:|:--------------------:|:--------------------:|
| KD | 73.33 | 72.98 | 73.54 | 74.92 | 70.66 | 73.08 | 70.67 |
| CTKD | 73.39 | 73.52 | 73.93 | 75.45 | 71.19 | 73.52 | 70.99 |
| DKD | 76.32 | 74.68 | 74.81 | 76.24 | 71.97 | 74.11 | 71.06 |
| MLKD | 77.08 | 75.18 | 75.35 | 76.63 | 72.19 | 74.11 | 71.89 |

### Training on ImageNet

- Download the dataset at <https://image-net.org/> and put it to `./data/imagenet`

  ```bash
  # KD
  python tools/train.py --cfg configs/imagenet/r34_r18/kd.yaml
  # KD+Ours
  python tools/train.py --cfg configs/imagenet/r34_r18/kd.yaml --logit-stand --base-temp 2 --kd-weight 9 
  ```

## Distilling ViTs

Please refer to [tiny-transformers](./tiny-transformers).

### Results

|    Model    |                      Top-1 Acc. (Base)                       |                      Top-1 Acc. (ECCV2022)                       | Tpo-1 Acc. (KD+Ours) |
| :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  DeiT-Tiny  | 65.08 ( [weights](https://drive.google.com/file/d/1UpnIPvcTWrBZ2FYCYYY4FkTK4LhXazUY/view?usp=sharing) \| [log](https://drive.google.com/file/d/1uAIoYeNPOIE141AO-95JnKUZqKPgtz3C/view?usp=sharing) ) | 78.15 ( [weights](https://drive.google.com/file/d/1vo8jugJkgxmgFtiS4V1tIKAfmg5jdh0D/view?usp=sharing) \| [log](https://drive.google.com/file/d/1agOqk8eIGK3_XqfNnLPKOKwDbKBeqffu/view?usp=sharing) ) | 78.55( [weights](https://drive.google.com/file/d/172r35OWaXFXopjQJNy5X0JRSLTtearvv/view?usp=sharing) \| [log](../logs/tiny-transformer/deit-ti_c100_KD_ours_3_10.txt)) |
|  T2T-ViT-7  | 69.37 ( [weights](https://drive.google.com/file/d/1walDSuqyy2zfQv55NuG9a8Eq5d3GlRuf/view?usp=sharing) \| [log](https://drive.google.com/file/d/17xsso8wUlt-cf_-oZavTn9i-c-pTMhUW/view?usp=sharing) ) | 78.35 ( [weights](https://drive.google.com/file/d/1wD3wQ13O7otXjRo-4dC9DHg_HdLoUTVT/view?usp=sharing) \| [log](https://drive.google.com/file/d/1SNILqkf18lX-qcKdkg200ZBYB3N-bOue/view?usp=sharing) ) |78.43( [weights](https://drive.google.com/file/d/1W6zQvIHa9EwvJXb9dwGsgjSTBvMANfIK/view?usp=sharing) \| [log](../logs/tiny-transformer/t2t-7_c100_KD_ours_3_6.txt)) |
|  PiT-Tiny   | 73.58 ( [weights](https://drive.google.com/file/d/1bTG9W0Kf-xNJSA35xv-Wmiw6G1Bfts3m/view?usp=sharing) \| [log](https://drive.google.com/file/d/1qhRMRp-AqBSFLvspHEsM06ANf8p6STox/view?usp=sharing) ) | 78.48 ( [weights](https://drive.google.com/file/d/14dPs5CzhVKqTwuwK3n75C-SWiWa3IQ6A/view?usp=sharing) \| [log](https://drive.google.com/file/d/1zYK9i9YN2mV9GMM02nbPRMOOGwqvehJg/view?usp=sharing) ) |78.76( [weights](https://drive.google.com/file/d/1sBy44PZt0Hn-24Xh3cYEwIIThTU9lO-g/view?usp=sharing) \| [log](../logs/tiny-transformer/pit-ti_c100_KD_ours.txt)) |
|  PVT-Tiny   | 69.22 ( [weights](https://drive.google.com/file/d/18BbtQ3XF-_tzOB9BNbu04C-KDsHhrqmM/view?usp=sharing) \| [log](https://drive.google.com/file/d/1Qb3sOi0AuXl726hqxXCZSI7i-qH8_1YL/view?usp=sharing) ) | 77.07 ( [weights](https://drive.google.com/file/d/1rDFwcz3s1Irxk3FE4OhHks7qlzmoxM-w/view?usp=sharing) \| [log](https://drive.google.com/file/d/1FJ5ajTGN6zr0Eo12B8gW4XJ2FUIMSNoT/view?usp=sharing) ) |78.43( [weights](https://drive.google.com/file/d/1Ms-Vq5UEpZK1aSjQ3UZGSVJLFhGZdwLX/view?usp=sharing) \| [log](../logs/tiny-transformer/pvt-ti_c100_KD_ours.txt)) |

## Acknowledgement

We would like to thank the authors of [logit-standardization-KD](https://github.com/sunshangquan/logit-standardization-KD)] which has significantly accelerated the development of our book.
