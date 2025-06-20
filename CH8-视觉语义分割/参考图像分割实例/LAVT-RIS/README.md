# 面向指代图像分割的语言感知视觉Transformer

This repository provides the PyTorch implementation of LAVT for Referring Image Segmentation tasks.

---

## Network Architecture

![Pipeline Image](pipeline.jpg)

---

## Environment Setup

Recommended: Python 3.7, PyTorch 1.7.1, CUDA 10.2

```bash
conda create -n lavt python=3.7
conda activate lavt
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

---

## Dataset Preparation

1. Refer to the `./refer` directory for instructions to prepare and download RefCOCO, RefCOCO+, G-Ref, etc.
2. Download 2014 Train images from [COCO official website](https://cocodataset.org/#download) and extract to `./refer/data/images/mscoco/images`.

---

## Pretrained Models and Weights

1. Create the `./pretrained_weights` directory, download [Swin Transformer pretrained weights](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth) and place them inside.
2. Create the `./checkpoints` directory, download LAVT training weights (see table below) and place them inside.

| RefCOCO | RefCOCO+ | G-Ref (UMD) | G-Ref (Google) |
|:-------:|:--------:|:-----------:|:--------------:|
| [weights](https://drive.google.com/file/d/13D-OeEOijV8KTC3BkFP-gOJymc6DLwVT/view?usp=sharing) | [weights](https://drive.google.com/file/d/1B8Q44ZWsc8Pva2xD_M-KFh7-LgzeH2-2/view?usp=sharing) | [weights](https://drive.google.com/file/d/1BjUnPVpALurkGl7RXXvQiAHhA-gQYKvK/view?usp=sharing) | [weights](https://drive.google.com/file/d/1weiw5UjbPfo3tCBPfB8tu6xFXCUG16yS/view?usp=sharing) |

---

## Training

For 4 GPUs, the training command is as follows (create the model save directory in advance):

```bash
mkdir ./models/refcoco
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py --model lavt --dataset refcoco --model_id refcoco --batch-size 8 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth --epochs 40 --img_size 480 2>&1 | tee ./models/refcoco/output
```
For other datasets, refer to the original command line instructions.

---

## Testing

Example for RefCOCO:

```bash
python test.py --model lavt --swin_type base --dataset refcoco --split val --resume ./checkpoints/refcoco.pth --workers 4 --ddp_trained_weights --window12 --img_size 480
```
Adjust parameters for other datasets and splits.

---

## Model Performance

LAVT evaluation results on various datasets:

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

You can use `demo_inference.py` for custom image and text inference and visualization.

---

## License

This repository is for academic research only and strictly prohibited for commercial use.
