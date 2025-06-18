# UniMatch

本仓库为 UniMatch 的官方 PyTorch 实现，适用于半监督语义分割任务，支持 Pascal VOC 2012、Cityscapes、COCO 等数据集。

---

## 环境配置

推荐环境：Python 3.10+，PyTorch 1.12.1+，CUDA 11.3+

```bash
cd UniMatch
conda create -n unimatch python=3.10.4
conda activate unimatch
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

---

## 预训练模型与权重

- ResNet-50: [下载地址](https://drive.google.com/file/d/1mqUrqFvTQ0k5QEotk4oiOFyP6B9dVZXS/view?usp=sharing)
- ResNet-101: [下载地址](https://drive.google.com/file/d/1Rx0legsMolCWENpfvE2jUScT3ogalMO8/view?usp=sharing)
- Xception-65: [下载地址](https://drive.google.com/open?id=1_j_mE07tiV24xXOJw4XDze0-a0NAhNVi)

请将权重文件放入 `./pretrained/` 目录下：

```
├── ./pretrained
    ├── resnet50.pth
    ├── resnet101.pth
    └── xception.pth
```

---

## 数据集准备

- Pascal VOC: [JPEGImages](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | [SegmentationClass](https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing)
- Cityscapes: [leftImg8bit](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [gtFine](https://drive.google.com/file/d/1E_27g9tuHm6baBqcA7jct_jqcGA89QPm/view?usp=sharing)
- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip) | [val2017](http://images.cocodataset.org/zips/val2017.zip) | [masks](https://drive.google.com/file/d/166xLerzEEIbU7Mt1UGut-3-VN41FMUb1/view?usp=sharing)

请将数据集解压并整理为如下结构，并在配置文件中修改路径：

```
├── [Your Pascal Path]
    ├── JPEGImages
    └── SegmentationClass
    
├── [Your Cityscapes Path]
    ├── leftImg8bit
    └── gtFine
    
├── [Your COCO Path]
    ├── train2017
    ├── val2017
    └── masks
```

---

## 训练与测试

### 训练

```bash
# 使用 torch.distributed.launch
sh scripts/train.sh <num_gpu> <port>
# 推荐4卡复现论文结果，否则需调整学习率
```

如需在其他数据集或划分上训练，请修改 `scripts/train.sh` 中的 `dataset` 和 `split` 参数。

### 测试

训练完成后可根据脚本或配置进行模型测试与评估。

---

## 模型性能

### Pascal VOC 2012

| Method                      | 1/16 (92) | 1/8 (183) | 1/4 (366) | 1/2 (732) | Full (1464) |
| :-------------------------: | :-------: | :-------: | :-------: | :-------: | :---------: |
| SupBaseline                 | 45.1      | 55.3      | 64.8      | 69.7      | 73.5        |
| U<sup>2</sup>PL             | 68.0      | 69.2      | 73.7      | 76.2      | 79.5        |
| ST++                        | 65.2      | 71.0      | 74.6      | 77.3      | 79.1        |
| PS-MT                       | 65.8      | 69.6      | 76.6      | 78.4      | 80.0        |
| **UniMatch (Ours)**         | **75.2**  | **77.2**  | **78.8**  | **79.9**  | **81.2**    |

### Cityscapes

| ResNet-50                   | 1/16      | 1/8       | 1/4       | 1/2       | ResNet-101           | 1/16        | 1/8         | 1/4         | 1/2         |
| :-------------------------: | :-------: | :-------: | :-------: | :-------: | :------------------: | :---------: | :---------: | :---------: | :---------: |
| SupBaseline                 | 63.3      | 70.2      | 73.1      | 76.6      | SupBaseline          | 66.3        | 72.8        | 75.0        | 78.0        |
| U<sup>2</sup>PL             | 70.6      | 73.0      | 76.3      | 77.2      | U<sup>2</sup>PL      | 74.9        | 76.5        | 78.5        | 79.1        |
| **UniMatch (Ours)**         | 75.0      | 76.8      | 77.5      | 78.6      | **UniMatch (Ours)**  | 76.6        | 77.9        | 79.2        | 79.5        |

### COCO

| Method                      | 1/512 (232) | 1/256 (463) | 1/128 (925) | 1/64 (1849) | 1/32 (3697) |
| :-------------------------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| SupBaseline                 | 22.9        | 28.0        | 33.6        | 37.8        | 42.2        |
| PseudoSeg                   | 29.8        | 37.1        | 39.1        | 41.8        | 43.6        |
| PC<sup>2</sup>Seg           | 29.9        | 37.5        | 40.1        | 43.7        | 46.1        |
| **UniMatch (Ours)**         | 31.9        | 38.9        | 44.4        | 48.2        | 49.8        |

---

## 许可

本仓库代码仅供学术研究使用，禁止商业用途
