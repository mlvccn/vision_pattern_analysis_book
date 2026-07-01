# 基于膨胀三维卷积神经网络的动作识别

该方法来源于国际会议CVPR2071中“[Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750)“一文。

环境配置

此代码是为 PyTorch 0.3 编写的。版本 0.4 及更高版本可能会出现问题。


# Fine-tuning and Feature Extraction
我们提供用于提取I3D特征并针对Charades任务进行微调的代码。我们的Charades微调模型也可在模型目录中获取（除DeepMind训练的模型外）。DeepMind预训练模型已转换为PyTorch格式，结果完全相同（flow_imagenet.pt和rgb_imagenet.pt）。这些模型是在ImageNet和Kinetics数据集上预训练的（详情请见[Kinetics-I3D](https://github.com/deepmind/kinetics-i3d)）。

## Fine-tuning I3D
[train_i3d.py](train_i3d.py) 包含了根据论文细节并由作者提供进行 I3D 微调的代码。具体而言，此版本遵循了作者在 2017 年 Charades 挑战赛中获胜的实现方式，对 [Charades](allenai.org/plato/charades/) 数据集进行微调。我们微调后的 RGB 和 Flow I3D 模型已存放在模型目录中（rgb_charades.pt 和 flow_charades.pt）。
这依赖于将光流和RGB帧提取并保存为图像到dist中。[charades_dataset.py](charades_dataset.py) 包含了用于加载训练视频片段的代码。

## Feature Extraction
[extract_features.py](extract_features.py) 包含加载预训练的 I3D 模型并提取特征，然后将特征保存为 numpy 数组的代码。[charades_dataset_full.py](charades_dataset_full.py) 脚本用于加载整个视频，以提取每个片段的特征。

## 致谢

我们想要感谢[I3D](https://github.com/piergiaj/pytorch-i3d)的作者，他们的工作显著加快了我们书籍的开发进程。
