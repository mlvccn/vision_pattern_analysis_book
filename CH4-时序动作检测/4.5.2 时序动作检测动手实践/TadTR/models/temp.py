import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
import math

class TadTR(nn.Module):
    """ TadTR 模型架构 """

    def __init__(self, position_embedding, transformer, num_classes, num_queries, aux_loss=True, with_segment_refine=True):
        super().__init__()
        self.num_queries = num_queries  # 查询数量，即最大可检测动作数
        self.transformer = transformer   # 使用的 Transformer 架构
        hidden_dim = transformer.d_model # 隐藏层维度

        # 分类头：将隐藏状态映射到类别得分
        self.class_embed = nn.Linear(hidden_dim, num_classes)

        # 回归头：使用 MLP 预测段坐标（中心点和宽度）
        self.segment_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        # 查询嵌入：每个查询由嵌入向量表示，包含位置信息
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)

        # 输入投影层：将输入特征转换为 Transformer 的维度
        self.input_proj = nn.Sequential(
            nn.Conv1d(2048, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim)
        )

        # 位置编码模块
        self.position_embedding = position_embedding

        # 是否启用辅助损失
        self.aux_loss = aux_loss

        # 是否启用迭代式段细化
        self.with_segment_refine = with_segment_refine

        # 初始化分类头偏置，使初始预测偏向无目标
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # 初始化回归头最后一层权重为 0
        nn.init.constant_(self.segment_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.segment_embed.layers[-1].bias.data, 0)

        # 初始化输入卷积层参数
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # 获取解码器层数
        num_pred = transformer.decoder.num_layers

        # 如果启用段细化，则克隆多个分类/回归头用于每层输出
        if with_segment_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.segment_embed = _get_clones(self.segment_embed, num_pred)
            # 偏置初始化，帮助模型更早预测出合理段长度
            nn.init.constant_(self.segment_embed[0].layers[-1].bias.data[1:], -2.0)
            self.transformer.decoder.segment_embed = self.segment_embed
        else:
            # 否则共享同一组分类/回归头
            nn.init.constant_(self.segment_embed.layers[-1].bias.data[1:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.segment_embed = nn.ModuleList([self.segment_embed for _ in range(num_pred)])
            self.transformer.decoder.segment_embed = None

    def forward(self, samples):
        """ 前向传播 """
        # 将输入统一为 NestedTensor 格式
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        # 获取位置编码
        pos = [self.position_embedding(samples)]

        # 提取输入张量与掩码
        src, mask = samples.tensors, samples.mask

        # 输入经过投影层进入 Transformer
        srcs = [self.input_proj(src)]
        masks = [mask]

        # 查询嵌入
        query_embeds = self.query_embed.weight

        # Transformer 主体计算
        hs, init_reference, inter_references, memory = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []

        # 收集每层解码器输出
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            # 当前层分类结果
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.segment_embed[lvl](hs[lvl])

            # 残差连接参考点
            if reference.shape[-1] == 2:
                tmp += reference
            else:
                tmp[..., 0] += reference[..., 0]

            # 归一化段坐标（中心 + 宽度）
            outputs_coord = tmp.sigmoid()

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        # 合并所有层输出
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        # 输出最终结果
        out = {
            'pred_logits': outputs_class[-1],
            'pred_segments': outputs_coord[-1]
        }

        # 若启用辅助损失，添加中间层输出
        if self.aux_loss:
            out['aux_outputs'] = [{'pred_logits': a, 'pred_segments': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        return out