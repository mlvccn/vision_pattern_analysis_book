# Squeeze-enhanced axial Transformer

本仓库为 SeaFormer 的官方 PyTorch 实现，适用于移动端语义分割任务，支持 ADE20K、Cityscapes 等数据集。

---

## 环境配置

推荐环境：Python 3.8+，PyTorch 1.8+，CUDA 11.1+

```bash
conda create -n seaformer python=3.8 -y
conda activate seaformer
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

---

## 数据集准备

请根据实际需求下载并准备 ADE20K、Cityscapes 等数据集，放置于 `data/` 目录下。数据集结构可参考 mmsegmentation 官方文档。

---

## 预训练模型与权重

- 分类模型权重与配置：[链接](seaformer-cls/)
- 语义分割模型权重与配置：[链接](seaformer-seg/)

---

## 训练与测试

### 语义分割训练示例

```bash
# 以 Cityscapes 为例
python tools/train.py --config seaformer-seg/cityscapes/seaformer_small.py
```

### 语义分割测试示例

```bash
python tools/test.py --config seaformer-seg/cityscapes/seaformer_small.py --checkpoint [权重路径]
```

---

## 模型性能

### ImageNet-1K 分类

| Model            | Size | Acc@1 | #Params (M) | FLOPs (G) |
|------------------|:----:|:-----:|:-----------:|:---------:|
| SeaFormer-Tiny   |  224 |  68.1 |     1.8     |    0.1    |
| SeaFormer-Small  |  224 |  73.4 |     4.1     |    0.2    |
| SeaFormer-Base   |  224 |  76.4 |     8.7     |    0.3    |
| SeaFormer-Large  |  224 |  79.9 |     14.0    |    1.2    |

### ADE20K 语义分割

| Method       |      Backbone    |   Pretrain  | Iters | mIoU(ss) |
|--------------|------------------|-------------|-------|----------|
|  Light Head  | SeaFormer-Tiny   | ImageNet-1K | 160K  | 36.5     |
|  Light Head  | SeaFormer-Small  | ImageNet-1K | 160K  | 39.4     |
|  Light Head  | SeaFormer-Base   | ImageNet-1K | 160K  | 41.9     |
|  Light Head  | SeaFormer-Large  | ImageNet-1K | 160K  | 43.8     |

### Cityscapes 语义分割

| Method         |      Backbone    |   FLOPs | mIoU |
|----------------|------------------|---------|----------|
|  Light Head(h) | SeaFormer-Small  |   2.0G  | 71.1     |
|  Light Head(f) | SeaFormer-Small  |   8.0G  | 76.4     |
|  Light Head(h) | SeaFormer-Base   |   3.4G  | 72.2     |
|  Light Head(f) | SeaFormer-Base   |   13.7G | 77.7     |

---

## 许可

本仓库代码仅供学术研究使用，禁止商业用途。

---

## 致谢

We would like to thank the authors of [Seaformer](https://github.com/fudan-zvg/SeaFormer)which has significantly accelerated the development of our book.

