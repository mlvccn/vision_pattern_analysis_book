# Dataset Distillation的复现指南

> 📄 论文地址：[PTQ4DiT: Post-training Quantization for Diffusion Transformers](https://arxiv.org/abs/1811.10959) （ICLR 2020）

## 环境要求

- Python 3
- CPU 或 NVIDIA GPU（支持 CUDA）

### 依赖库安装

- `torch >= 1.0.0`
- `torchvision >= 0.2.1`
- `numpy`
- `matplotlib`
- `pyyaml`
- `tqdm`

你可以参考 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择合适的安装方式。

## 基础使用：基本数据蒸馏（Basic Distillation）

希望将一个完整训练集中的知识压缩为极少的 *合成图像*（distilled images），这些图像用于训练新初始化的网络，仅需少量梯度更新步骤即可获得良好性能。

### 一、未知初始化（推荐）

每轮蒸馏迭代时会重新随机初始化模型权重。训练出的合成图像可以泛化至相似初始化分布下的模型。

- **MNIST**：

```bash
python main.py --mode distill_basic --dataset MNIST --arch LeNet
```

- **CIFAR-10**：

```bash
python main.py --mode distill_basic --dataset Cifar10 --arch AlexCifarNet \
    --distill_lr 0.001
```

> 注：`AlexCifarNet` 为来自 [cuda-convnet 项目](https://code.google.com/p/cuda-convnet2/) 的改编网络结构。

---

### 二、已知固定初始化（更高准确率）

此模式假设已知网络的初始化，可以用更少的合成图像实现更高准确率。

- **MNIST**：

```bash
python main.py --mode distill_basic --dataset MNIST --arch LeNet \
    --distill_steps 1 --train_nets_type known_init --n_nets 1 \
    --test_nets_type same_as_train
```

- **CIFAR-10**：

```bash
python main.py --mode distill_basic --dataset Cifar10 --arch AlexCifarNet \
    --distill_lr 0.001 --train_nets_type known_init --n_nets 1 \
    --test_nets_type same_as_train
```

---
