# 推拉学习方案（Pull-Push Learning Scheme, PPS）

该典型方法来源于国际会议CVPR2022"[Gaussian Mixture Proposals with Pull-Push Learning Scheme to Capture Diverse Events for Weakly Supervised Temporal Video Grounding](https://arxiv.org/pdf/2312.16388v1.pdf)"一文。

## 1.所需环境

```
- ubuntu 18.04.6
- cuda 11.6
- cudnn 8
- python 3.10.8
- pytorch 2.0.1
- nltk 3.8.1
- wandb 0.15.2
- h5py 3.8.0
- fairseq 0.12.2

若安装 `fairseq` 后自动安装了其他版本的 PyTorch，请删除后手动安装 `pytorch 2.0.1`。

NLTK资源下载：

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

## 2.数据准备

请准备以下特征文件，并按照如下目录结构放置：

```
data
├── activitynet
│   ├── sub_activitynet_v1-3.c3d.hdf5   # C3D特征（来源：https://github.com/JonghwanMun/LGI4temporalgrounding）
│   ├── glove.pkl
│   ├── train_data.json
│   ├── test_data.json
├── charades
│   ├── i3d_features.hdf5               # I3D特征（来源：https://github.com/minghangz/cpl）
│   ├── glove.pkl
│   ├── train.json
│   ├── test.json

```

## 3.模型测试（预训练模型评估）

首先从以下链接下载训练好的模型并放置到 `checkpoint/` 目录下：

[下载链接（Google Drive）](https://drive.google.com/file/d/1bf9d_UpnPSDZnmAmjD_E5o2SOaWOXDRg/view?usp=sharing)

### ActivityNet Captions 数据集：

- 原始代码版本 PPS：

```bash
bash script/eval_activitynet.sh
```

* 重构代码版本 PPS_re：

```bash
bash script/eval_activitynet_refact.sh
```

### Charades-STA 数据集：

- 原始代码版本 PPS：

```bash
bash script/eval_charades.sh
```

* 重构代码版本 PPS_re：

```bash
bash script/eval_charades_refact.sh
```

## 4.模型训练（从头训练）

### 在 ActivityNet Captions 数据集上训练：

在 ActivityNet Captions 数据集上训练：

```bash
bash script/train_activitynet.sh
```

在 Charades-STA 数据集上训练：

```bash
bash script/train_charades.sh
```

训练日志默认保存在 `log/` 文件夹，模型权重保存在 `checkpoint/` 文件夹。

使用 `wandb` 可视化训练过程，如需关闭，在 `config/` 目录下的配置文件中设置 `use_wandb=False`。

## 致谢

我们想要感谢[PPS](https://github.com/sunoh-kim/pps)的作者们，他们的工作显著加快了我们书籍的开发进程。
