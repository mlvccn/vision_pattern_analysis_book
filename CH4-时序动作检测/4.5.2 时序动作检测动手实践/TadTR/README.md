# 代码实践题

该方法来源于国际期刊TIP2022的 [End-to-end temporal action detection with Transformer](https://arxiv.org/abs/2106.10271) 一文。

## 题目要求

在时序动作检测任务中，Transformer 模型凭借其强大的全局建模能力，已成为主流结构之一。其自注意力机制能捕捉长距离依赖关系，有利于建模复杂的动作结构。然而，Transformer 对短时间窗口内的密集动作（如快速挥手、跳跃、快速转身等）往往建模不充分，原因在于其全局注意力可能稀释局部上下文细节。为了解决这一问题，可以考虑将局部建模机制引入 Transformer 框架，以提升模型对时间邻近帧之间高频变化的感知能力，从而改善细粒度动作检测性能。

**数据集**：THUMOS14数据集
**评价指标**：在指定的IoU（Intersection over Union）阈值下，计算出平均精度（Average Precision, AP）在所有类别上的平均值（即mAP@X，X表示给定阈值）以及在多个 IoU 阈值下（如 0.3~0.7）计算的mAP值的平均（即Avg.mAP）。

请你基于已有的TadTR模型代码，实现一个滑动窗口式局部自注意力模块，以进一步提升原模型的局部感知能力，改善模型对短时密集动作的检测效果，在THUMOS14数据集上进行训练和测试，然后使用给定的评价指标进行评估，观察检测精度相比原模型是否有所提升。

## 安装

### 要求

* Linux 或 Windows
  
* Python>=3.7

* (可选) CUDA>=9.2, GCC>=5.4
  
* PyTorch>=1.5.1, torchvision>=0.6.1（按照[此处](https://pytorch.org/)的说明安装）
  
* 其他依赖项
    ```bash
    pip install -r requirements.txt
    ```

### 编译 CUDA 扩展

RoIAlign 运算符使用了 CUDA 扩展实现，请使用有GPU的服务器进行训练。
```bash
cd model/ops;

# 如果你有多版本 CUDA 工具包安装，建议添加前缀
# CUDA_HOME=<你的CUDA工具包路径> 来指定正确的版本。 
python setup.py build_ext --inplace
```

## 数据准备

### THUMOS14

从 [[百度网盘(提取码: adTR)]](https://pan.baidu.com/s/183VprlbKNjMb3Gr-rfmROQ) 或 [[OneDrive]](https://husteducn-my.sharepoint.com/:f:/g/personal/liuxl_hust_edu_cn/EsMyXDlkrTdBsikoRQSIeUsBkxJJRsplbMyIQVYotiZRIQ?e=QYgiCH) 下载所有数据。

- **特征**：下载 I3D 特征 `I3D_2stream_Pth.tar`，此处已经将 RGB 和 Flow 特征进行了拼接（如果长度不一致则截断较长的一个），并将数据转换为 float32 精度以节省空间。
- **注释**：动作实例的注释和特征文件的元信息，两者都是 JSON 格式 (`th14_annotations_with_fps_duration.json` 和 `th14_i3d2s_ft_info.json`)。
- **预训练参考模型**：预训练模型使用了 I3D 特征 `thumos14_i3d2s_tadtr_reference.pth`，此模型对应配置文件 `configs/thumos14_i3d2s_tadtr.yml`。

下载完成后，通过以下命令解压归档的特征文件：
```bash
cd data
tar -xf I3D_2stream_Pth.tar
```

然后将特征、注释和模型放在 `data/thumos14` 目录下，期望根目录下的结构如下：

```
- data
  - thumos14
    - I3D_2stream_Pth
     - xxxxx
     - xxxxx
    - th14_annotations_with_fps_duration.json
    - th14_i3d2s_ft_info.json
    - thumos14_tadtr_reference.pth
```

## 测试预训练模型

运行以下命令：

```
python main.py --cfg CFG_PATH --eval --resume CKPT_PATH
```

其中 `CFG_PATH` 是定义实验设置的 YAML 格式配置文件的路径，例如 `configs/thumos14_i3d2s_tadtr.yml`。`CKPT_PATH` 是预训练模型的路径。或者，可以通过执行 Shell 脚本：

```
bash scripts/test_reference_models.sh thumos14
```

## 编写基于滑动窗口的局部自注意力模块

下面的代码位于`models/local_attention.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 新增的局部自注意力模块

class LocalContextAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, window_size=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.window_size = window_size  # 局部窗口大小（可配置）

        # 局部注意力层（每个窗口内的自注意力）
        self.local_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # 投影层（对齐维度）
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        B, T, C = x.shape
        # 生成滑动窗口索引（覆盖所有可能的局部窗口）
        # 例如：T=10, window_size=5 → 窗口索引为 [0-4, 1-5, ..., 5-9]
        window_indices = torch.tensor([
            list(range(i, i + self.window_size)) for i in range(T - self.window_size + 1)
        ], device=x.device)
        num_windows = window_indices.shape[0]

        # 从输入中提取所有窗口的特征（B, num_windows, window_size, C）
        windows = x[:, window_indices, :]  # 形状：[B, num_windows, window_size, C]

        # 对每个窗口计算自注意力（B, num_windows, window_size, C）
        # 调整形状为 (B*num_windows, window_size, C) 以适配MultiheadAttention的3维输入要求
        B, num_windows, window_size, C = windows.shape
        windows_reshaped = windows.view(B * num_windows, window_size, C)
        attn_output_reshaped, _ = self.local_attn(windows_reshaped, windows_reshaped, windows_reshaped)
        # 恢复为原始形状 (B, num_windows, window_size, C)
        attn_output = attn_output_reshaped.view(B, num_windows, window_size, C)

        # 将局部注意力结果合并回原时序维度（B, T, C）
        # 初始化输出矩阵（用0填充，后续累加）
        merged = torch.zeros_like(x)
        # 遍历每个窗口，将注意力结果累加到对应位置
        for i in range(num_windows):
            # 当前窗口覆盖的时间步：i 到 i + window_size - 1
            merged[:, i:i + self.window_size, :] += attn_output[:, i, :, :]

        # 投影并与原始特征相加（残差连接，保留原始信息）
        return x + self.proj(merged)
```

## 训练

运行以下命令：

```
python main.py --cfg CFG_PATH
```

**在 GPU 上运行**：由于模型非常轻量，只需一个 GPU 就足够了。可以通过在上述命令前添加前缀 `CUDA_VISIBLE_DEVICES=ID` 来指定使用的 GPU 设备 ID（例如 0）。要在多个 GPU 上运行，请参考 `scripts/run_parallel.sh`。

在训练期间，代码会每隔 N 个 epoch 自动进行测试（N 是 `opts.py` 中的 `test_interval`）。可以使用 Tensorboard 监控训练过程（需要在 `opts.py` 中设置 `cfg.tensorboard=True`），Tensorboard 记录和检查点将保存在 `output_dir`（可以在配置文件中修改）。

训练完成后，可以通过运行以下命令来测试训练好的模型：

```
python main.py --cfg CFG_PATH --eval
```

上面的代码会自动使用最佳模型检查点，如果想手动指定模型检查点，请运行如下代码：

```
python main.py --cfg CFG_PATH --eval --resume CKPT_PATH
```

## 实验结果

| 模型  | 特征提取器  | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Avg. mAP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TadTR | I3D 2stream | 73.8 | 68.7 | 59.5 | 45.2 | 32.4 | 55.9 |

## 致谢

我们想要感谢 [TadTR](https://github.com/xlliu7/TadTR) 的作者们，他们的工作显著加快了我们书籍的开发进程。
