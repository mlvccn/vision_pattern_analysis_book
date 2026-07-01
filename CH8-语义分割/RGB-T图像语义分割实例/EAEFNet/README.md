# 显式注意力增强融合的RGB-热成像感知任务方法

## Introduction

This repository provides the PyTorch implementation of EAEFNet: Explicit Attention-Enhanced Fusion for RGB-Thermal Perception Tasks. Parts of the code are adapted from [MFNet](https://github.com/haqishen/MFNet-pytorch), [RTFNet](https://github.com/yuxiangsun/RTFNet), [LSNet](https://github.com/zyrant/LSNet), and [RGBT-CC](https://github.com/chen-judge/RGBTCrowdCounting). The main branch supports **PyTorch 1.10+**.

---

## Installation

**Segmentation**

```shell
conda create -n EAEF python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate EAEF
cd EAEFNet_Seg_MF/EAEFNet_MF
pip install -r requirements.txt
```

---

## Dataset Download

- MFTNet Dataset: http://gofile.me/4jm56/CfukComo1
- PST900 Dataset: https://drive.google.com/file/d/1hZeM-MvdUC_Btyok7mdF00RV-InbAadm/view
You can also download from [Google Drive Collection](https://drive.google.com/drive/folders/1fqNwaumH0BrcAIvS0ebAjS35LX31Yw4S?usp=share_link).

---

## Pretrained Model Download

| Task   | Dataset | Model   | mIoU   | Weights Link                                               |
| ------ | ------- | ------- | ------ | ---------------------------------------------------------- |
| Seg    | MFNet   | EAEFNet | 58.91% | https://drive.google.com/drive/folders/12ONwVaaO35VbW7rZ83P-pSVWp_bFiPhv?usp=share_link |
| Seg    | PSP900  | EAEFNet | 85.56% | https://drive.google.com/drive/folders/1Czm7vtmaW6fTCk4fBAfO2OAWoHrJry9Z?usp=share_link |

---

## Testing

### MFNet Segmentation
```shell
cd EAEF_Seg_MF
python EAEF_MF/run_own_pth.py
```
### PST900 Segmentation
```shell
cd EAEF_Seg_PST
python EAEF_PST/run_own_pth.py
```

---

## Acknowledgement

We would like to thank the authors of [EAEFNet](https://github.com/FreeformRobotics/EAEFNet/tree/master) which has significantly accelerated the development of our book.


