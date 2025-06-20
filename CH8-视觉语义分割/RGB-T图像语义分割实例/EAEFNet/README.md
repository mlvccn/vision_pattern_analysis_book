# EAEFNet

## 简介

本仓库为 EAEFNet: Explicit Attention-Enhanced Fusion for RGB-Thermal Perception Tasks 的 PyTorch 实现。部分代码参考自 [MFNet](https://github.com/haqishen/MFNet-pytorch)、[RTFNet](https://github.com/yuxiangsun/RTFNet)、[LSNet](https://github.com/zyrant/LSNet) 和 [RGBT-CC](https://github.com/chen-judge/RGBTCrowdCounting)。主分支适配 **PyTorch 1.10+**。

---

## 安装

**Segmentation**

```shell
conda create -n EAEF python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate EAEF
cd EAEFNet_Seg_MF/EAEFNet_MF
pip install -r requirements.txt
```



---

## 数据集下载

- MFTNet Dataset: http://gofile.me/4jm56/CfukComo1
- PST900 Dataset: https://drive.google.com/file/d/1hZeM-MvdUC_Btyok7mdF00RV-InbAadm/view
也可从 [Google Drive合集](https://drive.google.com/drive/folders/1fqNwaumH0BrcAIvS0ebAjS35LX31Yw4S?usp=share_link) 下载。

---

## 预训练模型下载

| 任务         | 数据集 | 模型   | mIoU   | 训练权重                                               |
| ------------ | ------- | ------- | ------ | ------------------------------------------------------ |
| 分割         | MFNet   | EAEFNet | 58.91% | https://drive.google.com/drive/folders/12ONwVaaO35VbW7rZ83P-pSVWp_bFiPhv?usp=share_link |
| 分割         | PSP900  | EAEFNet | 85.56% | https://drive.google.com/drive/folders/1Czm7vtmaW6fTCk4fBAfO2OAWoHrJry9Z?usp=share_link |


---

## 测试方法


### MFNet 分割
```shell
cd EAEF_Seg_MF
python EAEF_MF/run_own_pth.py
```
### PST900 分割
```shell
cd EAEF_Seg_PST
python EAEF_PST/run_own_pth.py
```

---

## 致谢

We would like to thank the authors of [EAEFNet](https://github.com/FreeformRobotics/EAEFNet/tree/master) which has significantly accelerated the development of our book.


