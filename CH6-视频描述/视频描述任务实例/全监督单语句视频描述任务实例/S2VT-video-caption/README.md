# 序列到序列-视频到文本

该典型方法来源于国际会议ICCV2015的“[Sequence to Sequence -- Video to Text](https://arxiv.org/abs/1505.00487)”一文。

## 环境要求

本项目使用 Python 3.6。推荐使用 Anaconda 安装 PyTorch 和其他依赖库。

所需依赖：
- PyTorch
- Numpy
- tqdm
- pretrainedmodels
- tensorboardX
- TensorBoard
- ffmpeg（可通过 anaconda 安装）

## 数据

- MSVD 数据集：

  可以从以下链接下载 MSVD 数据集（提取码：s52p）：
  https://pan.baidu.com/s/1-NcMX7TLHrm4wN8bnbED7w

  或者直接下载已经提取好的特征（使用的是带有 BatchNorm 的 VGG16，提取码：muph）：
  https://pan.baidu.com/s/1aQw9OZ3U8g3RQ4RsmL74RQ

  可以使用 `extract_features.py` 脚本提取你自己的特征。

  下载或提取完特征后，请在 `train.py` 中修改路径。

## 使用方法

步骤1：安装所有所需依赖包。

步骤2：下载 MSVD 数据集（或后续支持的 MSR-VTT）。

步骤3：提取特征

  运行以下命令：
  python extract_features.py --video_path ./data/YouTubeClips --feat_path ./data/feats/msvd_vgg16_bn --model vgg16_bn

  （可能需要先创建 feat_path 目录，详情见 extract_features.py）

步骤4：准备描述数据

  运行 prepare_captions.py。可能需要在脚本中修改数据路径配置。

步骤5：模型训练

  运行 train.py。训练参数可以通过类 Opt() 进行配置。

步骤6：测试 / 评估

  运行 eval.py。可以选择使用 Beam Search 或常规方法进行生成。

  本项目提供的评估指标包括：
  - Bleu_1、Bleu_2、Bleu_3、Bleu_4
  - METEOR
  - ROUGE_L
  - CIDEr

步骤7：查看日志

  可以在 ./runs 中查看 tensorboard 日志。运行命令：

  tensorboard --logdir=./runs

  日志中包含学习率、损失值和各层权重信息。

  训练时的配置信息也会以 .txt 文件形式保存在 ./checkpoint 目录下。

## 结果

| 模型                             | 网络结构 / 特征    | METEOR (%) |
| ------------------------------ | ------------ | ---------- |
| FGM                            | 模板生成         | 23.9       |
| Mean Pool - AlexNet            | 平均池化         | 26.9       |
| Mean Pool - VGG                | 平均池化         | 27.7       |
| Mean Pool - AlexNet + COCO     | 平均池化 + 预训练   | 29.1       |
| Mean Pool - GoogleNet          | 平均池化         | 28.7       |
| Temporal Attention - GoogleNet | 帧加权注意力       | 29.0       |
| Temp. Attention + 3D-CNN       | 三维卷积网络 + 注意力 | 29.6       |
| **S2VT - Flow (AlexNet)**      | 光流输入         | 24.3       |
| **S2VT - RGB (AlexNet)**       | 原始帧输入        | 27.9       |
| **S2VT - RGB (VGG)**           | 原始帧 + 顺序训练   | 29.2       |
| **S2VT - RGB (VGG, 打乱顺序)**  | 原始帧 + 随机顺序   | 28.2       |
| **S2VT - RGB (VGG) + Flow**    | RGB + 光流（融合） | **29.8**   |


## 致谢

我们想要感谢 [S2VT](https://github.com/vsubhashini/caffe/tree/recurrent/examples/s2vt) 的作者们，他们的工作显著加快了我们书籍的开发进程。