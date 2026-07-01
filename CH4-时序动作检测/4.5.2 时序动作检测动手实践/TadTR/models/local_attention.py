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