# 基于纯文本训练解码 CLIP 隐状态的零样本描述方法

该方法来源于国际会议ICLR2023的“[DeCap: Decoding CLIP Latents for Zero-Shot Captioning via Text-Only Training](https://openreview.net/pdf?id=Lt8bMlhiwx2)”一文。

## 数据下载
将以下文件下载并放置在 `data` 目录下：

- coco_train: https://drive.google.com/file/d/1k4LlhgwnvpkUlzQjtTomnDFvlkboTxOH/view?usp=share_link
- cc3m_train: https://drive.google.com/file/d/1-xfOLJasBTqTrSnsyAncKSfsjSSN5RTH/view?usp=share_link

## 模型训练

实验在RTX 3090 GPU上运行

运行以下命令之一开始训练：

./train_coco.sh

或

./train_cc3m.sh

## 推理（Inference）
请参考 `inference_decap.ipynb` 文件，查看如何进行推理生成图像描述。在处理视频任务时请自行完成视频帧平均的操作

## 预训练模型
- 基于 COCO 数据集训练的模型: https://drive.google.com/file/d/1EFI0aujIWBr3dTC_a2hdoV4QJenAlEWU/view?usp=share_link

## 结果

| 模型名称          | CIDEr    | SPICE    | METEOR   | BLEU-4    | BLEU-1    |
| ------------- | -------- | -------- | -------- | --------- | --------- |
| CLIPRe-MSR    | 10.2     | 18.8     | 19.9     | **0.835** | **0.852** |
| DeCap-VD-MSR  | 5.9      | 16.3     | 10.2     | 0.765     | 0.722     |
| DeCap-NND-MSR | 13.1     | 20.2     | 24.4     | 0.805     | 0.771     |
| **DeCap-MSR** | **23.1** | **23.6** | **34.8** | 0.823     | 0.770     |


## 致谢
DeCap项目大量借鉴了 ClipCap (https://github.com/rmokady/CLIP_prefix_caption) 的代码框架。训练过程中使用了以下数据集：

- COCO dataset: https://cocodataset.org/#home
- Conceptual Captions: https://ai.google.com/research/ConceptualCaptions/

我们想要感谢[DeCap](https://github.com/dhg-wei/DeCap) 的作者们，他们的工作显著加快了我们书籍的开发进程。