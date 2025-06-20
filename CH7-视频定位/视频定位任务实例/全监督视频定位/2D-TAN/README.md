# 二维时序邻接网络（2D  Temporal Adjacent Networks, 2D-TAN）

该典型方法来源于国际会议AAAI2020“[Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language](https://github.com/microsoft/VideoX/tree/master/2D-TAN)”一文。

## 1.所需环境

```
- CUDA 11.7
- Python 3.7
- PyTorch 1.17.0
- 依赖库：
  - torchtext
  - easydict
  - terminaltables
```

## 2.数据准备

请从以下链接之一下载预提取的视频特征，并解压到 `data/` 目录下：

- [Box 链接](https://rochester.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav)
- [Dropbox 链接](https://www.dropbox.com/sh/dszrtb85nua2jqe/AABGAEQhPtqBIRpGPY3gZey6a?dl=0)

## 3.训练命令

以下为不同数据集和模型配置下的训练命令：

```bash
# Charades-STA
python moment_localization/train.py --cfg experiments/charades/2D-TAN-16x16-K5L8-pool.yaml --verbose
python moment_localization/train.py --cfg experiments/charades/2D-TAN-16x16-K5L8-conv.yaml --verbose

# ActivityNet Captions
python moment_localization/train.py --cfg experiments/activitynet/2D-TAN-64x64-K9L4-pool.yaml --verbose
python moment_localization/train.py --cfg experiments/activitynet/2D-TAN-64x64-K9L4-conv.yaml --verbose

# TACoS
python moment_localization/train.py --cfg experiments/tacos/2D-TAN-128x128-K5L8-pool.yaml --verbose
python moment_localization/train.py --cfg experiments/tacos/2D-TAN-128x128-K5L8-conv.yaml --verbose
```

## 4.测试命令

```bash
# Charades-STA
python moment_localization/test.py --cfg experiments/charades/2D-TAN-16x16-K5L8-pool.yaml --verbose --split test
python moment_localization/test.py --cfg experiments/charades/2D-TAN-16x16-K5L8-conv.yaml --verbose --split test

# ActivityNet Captions
python moment_localization/test.py --cfg experiments/activitynet/2D-TAN-64x64-K9L4-pool.yaml --verbose --split test
python moment_localization/test.py --cfg experiments/activitynet/2D-TAN-64x64-K9L4-conv.yaml --verbose --split test

# TACoS
python moment_localization/test.py --cfg experiments/tacos/2D-TAN-128x128-K5L8-pool.yaml --verbose --split test
python moment_localization/test.py --cfg experiments/tacos/2D-TAN-128x128-K5L8-conv.yaml --verbose --split test
```

## 致谢

我们想要感谢[2D-TAN](https://github.com/microsoft/VideoX/tree/master/2D-TAN)的作者们，他们的工作显著加快了我们书籍的开发进程。
