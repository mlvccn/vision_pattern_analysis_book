# EAEFNet

![license](https://img.shields.io/badge/license-MIT-green) ![PyTorch-1.10.0](https://img.shields.io/badge/PyTorch-1.10.0-blue)

## 简介

本仓库为 EAEFNet: Explicit Attention-Enhanced Fusion for RGB-Thermal Perception Tasks 的 PyTorch 实现，支持分割、检测、计数、显著性检测等多种 RGB-T 任务。部分代码参考自 [MFNet](https://github.com/haqishen/MFNet-pytorch)、[RTFNet](https://github.com/yuxiangsun/RTFNet)、[LSNet](https://github.com/zyrant/LSNet) 和 [RGBT-CC](https://github.com/chen-judge/RGBTCrowdCounting)。主分支适配 **PyTorch 1.10+**。

---

## 安装

**Crow-Counting**

```shell
conda create -n EAEF python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate EAEF
cd EAEFNet_RGBT-CC
pip install -r requirements.txt
```

**Detection**

```shell
conda create -n EAEF python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate EAEF
pip install openmim
mim install "mmengine>=0.3.1"
mim install "mmcv>=2.0.0rc1,<2.1.0"
mim install "mmdet>=3.0.0rc3,<3.1.0"
cd EAEFNet_Detection/EAEF_mmyolo
pip install -r requirements/albu.txt
mim install -v -e .
```

**Segmentation**

```shell
conda create -n EAEF python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate EAEF
cd EAEFNet_Seg_MF/EAEFNet_MF
pip install -r requirements.txt
```

**SOD**

```shell
conda create -n EAEF python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate EAEF
cd EAEFNet_SOD
pip install -r requirements.txt
```

---

## 数据集下载

- MFTNet Dataset: http://gofile.me/4jm56/CfukComo1
- PST900 Dataset: https://drive.google.com/file/d/1hZeM-MvdUC_Btyok7mdF00RV-InbAadm/view
- M3FD Dataset: https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6
- RGBT-CC Dataset: https://www.dropbox.com/sh/o4ww2f5tv3nay9n/AAA4CfVMTZcdwsFxFlhwDsSba?dl=0

也可从 [Google Drive合集](https://drive.google.com/drive/folders/1fqNwaumH0BrcAIvS0ebAjS35LX31Yw4S?usp=share_link) 下载。

---

## 预训练模型下载

| 任务         | 数据集 | 模型   | mIoU   | 训练权重                                               |
| ------------ | ------- | ------- | ------ | ------------------------------------------------------ |
| 分割         | MFNet   | EAEFNet | 58.91% | https://drive.google.com/drive/folders/12ONwVaaO35VbW7rZ83P-pSVWp_bFiPhv?usp=share_link |
| 分割         | PSP900  | EAEFNet | 85.56% | https://drive.google.com/drive/folders/1Czm7vtmaW6fTCk4fBAfO2OAWoHrJry9Z?usp=share_link |

| 任务      | 数据集 | 模型       | mAP@0.5 | 训练权重                                               |
| --------- | ------- | ----------- | ------- | ------------------------------------------------------ |
| 检测      | M3FD    | EAEF+Yolov5 | 80.4%   | https://drive.google.com/drive/folders/1JcvZUmTUB936H9JoYjYrM9H-jHKnjNzc?usp=share_link |

| 任务          | 数据集 | 模型 | RMSE   | 训练权重                                               |
| ------------- | ------- | ----- | ------ | ------------------------------------------------------ |
| 人群计数      | RGBTCC | EAEF  | 21.85% | https://drive.google.com/drive/folders/1eb0GwISb0AUULrDpUo8jBZC5Oh4zShgD?usp=share_link |

| 任务          | 数据集 | 模型 | MAE   | 训练权重                                               |
| ------------- | ------- | ----- | ------ | ------------------------------------------------------ |
| 显著性检测    | VT5000 | EAEF  | 0.0031 | [Google Drive](https://drive.google.com/drive/folders/1nxvFOOQN8a0U17hFGqcny8VgsGTPBBWz?usp=sharing) |

---

## 测试方法

### M3FD 检测
```shell
python EAEF_mmyolo/tools/test.py yolov5/bi_yolov5 
```
### MFNet 分割
```shell
python EAEF_MF/run_own_pth.py
```
### PST900 分割
```shell
python EAEF_PST/run_own_pth.py
```
### RGBT-CC 人群计数
```shell
python EAEF_CC/test.py
```
### VT821/VT1000/VT5000 显著性检测
```shell
python EAEF_SOD/test.py
```

---

## 许可

本仓库代码仅供学术研究使用，禁止商业用途。


