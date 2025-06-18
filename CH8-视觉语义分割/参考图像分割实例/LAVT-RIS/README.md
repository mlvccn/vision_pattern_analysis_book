# LAVT: Language-Aware Vision Transformer for Referring Image Segmentation

本仓库为 LAVT 的 PyTorch 实现，适用于 Referring Image Segmentation 任务。

---

## 网络结构

![Pipeline Image](pipeline.jpg)

---

## 环境配置

推荐环境：Python 3.7，PyTorch 1.7.1，CUDA 10.2

```bash
conda create -n lavt python=3.7
conda activate lavt
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

---

## 数据集准备

1. 参考 `./refer` 目录说明，准备并下载 RefCOCO、RefCOCO+、G-Ref 等数据集标注。
2. 从 [COCO官网](https://cocodataset.org/#download) 下载 2014 Train images，解压到 `./refer/data/images/mscoco/images`。

---

## 预训练模型与权重

1. 创建 `./pretrained_weights` 目录，下载 [Swin Transformer 预训练权重](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth) 并放入其中。
2. 创建 `./checkpoints` 目录，下载 LAVT 训练权重（见下表）并放入其中。

| RefCOCO | RefCOCO+ | G-Ref (UMD) | G-Ref (Google) |
|:-------:|:--------:|:-----------:|:--------------:|
| [权重](https://drive.google.com/file/d/13D-OeEOijV8KTC3BkFP-gOJymc6DLwVT/view?usp=sharing) | [权重](https://drive.google.com/file/d/1B8Q44ZWsc8Pva2xD_M-KFh7-LgzeH2-2/view?usp=sharing) | [权重](https://drive.google.com/file/d/1BjUnPVpALurkGl7RXXvQiAHhA-gQYKvK/view?usp=sharing) | [权重](https://drive.google.com/file/d/1weiw5UjbPfo3tCBPfB8tu6xFXCUG16yS/view?usp=sharing) |

---

## 训练

以 4 卡为例，训练命令如下（需提前创建模型保存目录）：

```bash
mkdir ./models/refcoco
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py --model lavt --dataset refcoco --model_id refcoco --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth --epochs 40 --img_size 480 2>&1 | tee ./models/refcoco/output
```
其他数据集训练命令类似，详见原始命令行参数说明。

---

## 测试

以 RefCOCO 为例：

```bash
python test.py --model lavt --swin_type base --dataset refcoco --split val --resume ./checkpoints/refcoco.pth --workers 4 --ddp_trained_weights --window12 --img_size 480
```
其他数据集和分割集可调整参数。

---

## 模型性能

LAVT 在各数据集上的评测结果如下：

|     Dataset     | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 | Overall IoU | Mean IoU |
|:---------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------:|:--------:|
| RefCOCO val     | 84.46 | 80.90 | 75.28 | 64.71 | 34.30 |    72.73    |   74.46  |
| RefCOCO test A  | 88.07 | 85.17 | 79.90 | 68.52 | 35.69 |    75.82    |   76.89  |
| RefCOCO test B  | 79.12 | 74.94 | 69.17 | 59.37 | 34.45 |    68.79    |   70.94  |
| RefCOCO+ val    | 74.44 | 70.91 | 65.58 | 56.34 | 30.23 |    62.14    |   65.81  |
| RefCOCO+ test A | 80.68 | 77.96 | 72.90 | 62.21 | 32.36 |    68.38    |   70.97  |
| RefCOCO+ test B | 65.66 | 61.85 | 55.94 | 47.56 | 27.24 |    55.10    |   59.23  |
| G-Ref val (UMD) | 70.81 | 65.28 | 58.60 | 47.49 | 22.73 |    61.24    |   63.34  |
| G-Ref test (UMD)| 71.54 | 66.38 | 59.00 | 48.21 | 23.10 |    62.09    |   63.62  |
|G-Ref val (Goog.)| 71.16 | 67.21 | 61.76 | 51.98 | 27.30 |    60.50    |   63.66  |

---

## Demo

可通过 `demo_inference.py` 脚本对自定义图片和文本进行推理与可视化。

---

## 许可

本仓库代码仅供学术研究使用，禁止商业用途
