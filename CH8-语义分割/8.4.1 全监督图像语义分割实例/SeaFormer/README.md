# 轴向压缩增强Transformer

This repository provides the official PyTorch implementation of SeaFormer, designed for mobile semantic segmentation tasks and supporting datasets such as ADE20K and Cityscapes.

---

## Environment Setup

Recommended: Python 3.8+, PyTorch 1.8+, CUDA 11.1+

```bash
conda create -n seaformer python=3.8 -y
conda activate seaformer
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

---

## Dataset Preparation

Please download and prepare ADE20K, Cityscapes, etc. as needed, and place them in the `data/` directory. The dataset structure can refer to the official mmsegmentation documentation.

---

## Pretrained Models and Weights

- Classification model weights and configs: [link](seaformer-cls/)
- Semantic segmentation model weights and configs: [link](seaformer-seg/)

---

## Training and Testing

### Semantic Segmentation Training Example

```bash
# Example for Cityscapes
python tools/train.py --config seaformer-seg/cityscapes/seaformer_small.py
```

### Semantic Segmentation Testing Example

```bash
python tools/test.py --config seaformer-seg/cityscapes/seaformer_small.py --checkpoint [weight_path]
```

---

## Model Performance

### ImageNet-1K Classification

| Model            | Size | Acc@1 | #Params (M) | FLOPs (G) |
|------------------|:----:|:-----:|:-----------:|:---------:|
| SeaFormer-Tiny   |  224 |  68.1 |     1.8     |    0.1    |
| SeaFormer-Small  |  224 |  73.4 |     4.1     |    0.2    |
| SeaFormer-Base   |  224 |  76.4 |     8.7     |    0.3    |
| SeaFormer-Large  |  224 |  79.9 |     14.0    |    1.2    |

### ADE20K Semantic Segmentation

| Method       |      Backbone    |   Pretrain  | Iters | mIoU(ss) |
|--------------|------------------|-------------|-------|----------|
|  Light Head  | SeaFormer-Tiny   | ImageNet-1K | 160K  | 36.5     |
|  Light Head  | SeaFormer-Small  | ImageNet-1K | 160K  | 39.4     |
|  Light Head  | SeaFormer-Base   | ImageNet-1K | 160K  | 41.9     |
|  Light Head  | SeaFormer-Large  | ImageNet-1K | 160K  | 43.8     |

### Cityscapes Semantic Segmentation

| Method         |      Backbone    |   FLOPs | mIoU |
|----------------|------------------|---------|----------|
|  Light Head(h) | SeaFormer-Small  |   2.0G  | 71.1     |
|  Light Head(f) | SeaFormer-Small  |   8.0G  | 76.4     |
|  Light Head(h) | SeaFormer-Base   |   3.4G  | 72.2     |
|  Light Head(f) | SeaFormer-Base   |   13.7G | 77.7     |

---

## License

This repository is for academic research only and strictly prohibited for commercial use.

---

## Acknowledgement

We would like to thank the authors of [Seaformer](https://github.com/fudan-zvg/SeaFormer) which has significantly accelerated the development of our book.

