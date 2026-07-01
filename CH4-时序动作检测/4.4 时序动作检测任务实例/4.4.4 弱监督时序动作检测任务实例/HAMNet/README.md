# 面向弱监督时序动作定位的混合注意力机制方法

该方法来源于国际会议AAAI2021的 [A hybrid attention mechanism for weakly-supervised temporal action localization](https://arxiv.org/abs/2101.00545) 一文。

## 安装

### 环境依赖

- PyTorch 1.7.1  
- pytorch-lightning 1.1.2  
- 还需安装 loguru、colorama 等依赖  

推荐使用以下命令创建 conda 环境（注意，请适配CUDA版本安装对应的pytorch等相关包，否则会报错）：

```bash
conda env create -f environment.yml
```

## 数据准备

从以下链接下载 THUMOS14 的 ground-truth 和 I3D 特征：[Box 数据链接](https://rpi.box.com/s/hf6djlgs7vnl7a2oamjt0vkrig42pwho)

下载完成后，将文件/文件夹放置在：

```
./data/
```

## 训练&测试

### 训练

执行以下命令开始训练：

```bash
python main.py
```

可通过 `options.py` 查看更多命令行参数选项。

### 模型测试

执行以下命令进行测试：

```bash
python main.py --test --ckpt [checkpoint_path]
```

将 `[checkpoint_path]` 替换为实际的模型路径。

## 实验结果

| 模型    | 数据集   | Avg.IoU(0.1:0.7) |
| ------- | -------- | ---------------- |
| HAM-Net | THUMOS14 | 39.7             |

## 致谢

我们想要感谢 [HAM-Net](https://github.com/asrafulashiq/hamnet) 的作者们，他们的工作显著加快了我们书籍的开发进程。
