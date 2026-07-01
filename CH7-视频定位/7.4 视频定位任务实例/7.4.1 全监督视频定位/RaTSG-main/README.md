# 关系感知时序语句定位网络（Relation-aware Temporal Sentence Grounding, RaTSG）
该典型方法来源于国际会议NeurIPS2024“[Temporal Sentence Grounding with Relevance Feedback in Videos](https://nips.cc/media/neurips-2024/Slides/94274.pdf#:~:text=We%20propose%20the%20Relation-aware%20Temporal%20Sentence%20Grounding%20%28RaTSG%29,fine-grained%20and%20coarse-grained%20relevance%20between%20text%20and%20video.)”一文。

## 1.环境要求

```
- Ubuntu 20.04  
- CUDA 11.7  
- Python 3.7  

安装依赖：
pip install -r requirements.txt
```

## 2.数据准备

数据集位于 `./data/dataset` 目录，包含改造后的验证集和测试集：

- Charades-STA-RF
- ActivityNet Captions-RF

### 特征准备

推荐直接下载准备好的视觉特征（包括 Charades 和 ActivityNet）并放置于 `./data/features/` 目录下：

- [视觉特征下载地址 (Mega)](https://mega.nz/folder/gv93jDSI#U9Qf1ZuKdP8cIJj5sdK0bw)

下载 GloVe 词向量（840B.300d）并放置于相同目录：

- [GloVe 下载地址](http://nlp.stanford.edu/data/glove.840B.300d.zip)

## 3.训练模型

```bash
# 训练 RaTSG 模型（Charades-STA-RF）
bash charades_RF_train.sh

# 训练 RaTSG 模型（ActivityNet Captions-RF）
bash activitynet_RF_train.sh
```

## 4.评估模型

```bash
# 测试 Charades-STA-RF 上的模型
bash charades_RF_test.sh

# 测试 ActivityNet Captions-RF 上的模型
bash activitynet_RF_test.sh
```

## 5.预训练模型下载

将以下模型文件下载并放置至 `./ckpt/` 目录：

- [RaTSG_charades_RF_i3d_128](https://drive.google.com/drive/folders/1TQyojFEEhXsDg6GSChfGrmCcoKesigI5?usp=sharing)
- [RaTSG_activitynet_RF_i3d_128](https://drive.google.com/drive/folders/1TQyojFEEhXsDg6GSChfGrmCcoKesigI5?usp=sharing)

## 6.实验结果

|     数据集     |  Acc  | R1@0.3 | R1@0.5 | R1@0.7 | mIoU  |
| :------------: | :---: | :----: | :----: | :----: | :---: |
|  Charades-RF   | 76.85 | 68.17  | 61.61  | 54.19  | 59.93 |
| ActivityNet-RF | 84.27 | 69.02  | 60.68  | 52.88  | 61.15 |



## 致谢

我们想要感谢[RaTSG](https://github.com/HuiGuanLab/RaTSG)的作者们，他们的工作显著加快了我们书籍的开发进程。
