# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from opts import cfg

# if not cfg.disable_cuda:
#     from .functions import TDAFunction

import warnings
import math
import pdb

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_



def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class DeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=1, n_heads=8, n_points=4):
        """
        可变形注意力模块
        :param d_model      隐藏维度
        :param n_levels     特征层级数
        :param n_heads      注意力头数
        :param n_points     每个注意力头的采样点数
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                'd_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # 建议将_d_per_head设置为2的幂次，这在我们的CUDA实现中效率更高
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in DeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        assert n_levels == 1, 'multi-level attention is not supported!'

        self.seq2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(
            d_model, n_heads * n_levels * n_points)
        self.attention_weights = nn.Linear(
            d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        # 初始偏移量:
        # (1, 0, -1, 0, -1, 0, 1, 0)
        thetas = torch.arange(
            self.n_heads, dtype=torch.float32) * (4.0 * math.pi / self.n_heads)
        grid_init = thetas.cos()[:, None]

        grid_init = grid_init.view(self.n_heads, 1, 1, 1).repeat(
            1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_temporal_lens, input_level_start_index, input_padding_mask=None):
        """
        :param query (= src + pos)         (N, 查询长度, C)
        :param reference_points            (N, 查询长度, 特征层级数, 1)，范围在[0, 1]，左边界(0)，右边界(1)，包含填充区域
                                        或(N, 查询长度, 特征层级数, 2)，添加额外的(t)形成参考片段
        :param input_flatten (=src)        (N, \sum_{l=0}^{L-1} T_l, C)
        :param input_temporal_lens         (特征层级数)，[T_0, T_1, ..., T_(L-1)]
        :param input_level_start_index     (特征层级数, )，[0, T_0, T_1, T_2, ..., T_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} T_l)，填充元素为True，非填充元素为False

        :return output                     (N, 查询长度, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert input_temporal_lens.sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads,
                           self.d_model // self.n_heads)
        # 时间轴上的预测偏移量。这些是*绝对*值，未归一化
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 1)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(
            attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 1:
            # the reference points are normalized, but the offset are unnormalized
            # so we need to normalize the offsets
            offset_normalizer = input_temporal_lens[..., None]
            # (N, Length_{query}, n_heads, n_levels, n_points, 1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
        # deform attention in the l-th (l >= 2) decoder layer when segment refinement is enabled
        elif reference_points.shape[-1] == 2:
            # offsets are related with the size of the reference segment
            sampling_locations = reference_points[:, :, None, :, None, :1] \
                + sampling_offsets / self.n_points * \
                reference_points[:, :, None, :, None, 1:] * 0.5

        else:
            raise ValueError(
                'Last dim of reference_points must be 1 or 2, but get {} instead.'.format(reference_points.shape[-1]))
        if cfg.dfm_att_backend == 'pytorch' or cfg.disable_cuda:
            # 使用PyTorch的grid_sample操作实现。
        # 注意grid_sample仅支持图像输入。我们需要将序列视为高度为1的图像
            sampling_locations = torch.cat((sampling_locations, torch.ones_like(sampling_locations)*0.5), dim=-1)
            input_spatial_shapes = torch.stack((torch.ones_like(input_temporal_lens), input_temporal_lens), dim=-1)
            output = deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        else:
            raise NotImplementedError
            # # CUDA implementation. You will get identical results with the pytorch implementation
            # output = TDAFunction.apply(
            #     value, input_temporal_lens, input_level_start_index, sampling_locations, attention_weights, self.seq2col_step)
        output = self.output_proj(output)
        return output, (sampling_locations, attention_weights)


def deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    '''deformable attention implemeted with grid_sample.'''
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()