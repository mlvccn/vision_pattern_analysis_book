# -*- coding: utf-8 -*-
import torch
import numpy as np
from utils.loss_spot import spot_loss, softmax_mse_loss, softmax_kl_loss  # 假设损失函数来自utils.loss_spot


def train_semi(data_loader, train_loader_unlabel, model, optimizer, epoch):
    """
    半监督训练主函数（精简版）
    Args:
        data_loader: 有标签数据加载器
        train_loader_unlabel: 无标签数据加载器
        model: 模型实例
        optimizer: 优化器
        epoch: 当前训练轮次
    """
    # 初始化参数
    model.train()  # 模型设为训练模式
    total_loss = 0
    consistency_criterion = softmax_mse_loss  # 一致性损失函数（MSE）
    consistency_criterion_top = softmax_kl_loss  # 顶部一致性损失函数（KL散度）
    temporal_perb = TemporalShift_random(400, 64)  # 时间维度随机移位增强
    unlabeled_train_iter = iter(train_loader_unlabel)  # 无标签数据迭代器

    # 关键超参数
    temp = 0.6  # 温度参数（用于软标签）
    u_thres = 0.95  # 无标签数据置信度阈值
    lambda_3 = 1.0  # 损失权重

    for n_iter, (input_data, top_br_gt, bottom_br_gt, action_gt, label_gt, _, _, _) in enumerate(data_loader):
        # ------------------------ 有标签数据处理 ------------------------
        # 数据增强：时间移位、翻转、Dropout
        input_data_shift = temporal_perb(input_data)  # 时间移位增强
        input_data_flip = input_data.flip(2).contiguous()  # 时间翻转增强
        input_data_shift = F.dropout2d(input_data_shift, 0.2)  # 空间Dropout
        input_data_flip = F.dropout2d(input_data_flip, 0.1)  # 空间Dropout

        # 前向传播：计算有标签数据预测结果
        top_br_pred, bottom_br_pred, feat = model(input_data.cuda())
        loss_label = spot_loss(top_br_gt, top_br_pred, bottom_br_gt, bottom_br_pred, action_gt, label_gt)  # 有标签损失

        # ------------------------ 无标签数据处理 ------------------------
        # 加载无标签数据（若迭代器耗尽则重置）
        try:
            input_data_unlabel = unlabeled_train_iter.next()[0].cuda()
        except StopIteration:
            unlabeled_train_iter = iter(train_loader_unlabel)
            input_data_unlabel = unlabeled_train_iter.next()[0].cuda()

        # 前向传播：计算无标签数据预测结果
        top_br_pred_unlabel, bottom_br_pred_unlabel, feat_unlabel = model(input_data_unlabel)

        # 动态阈值伪标签筛选
        pseudo_label = torch.softmax(top_br_pred_unlabel, dim=1)
        max_probs, max_idx = torch.max(pseudo_label, dim=1)
        mask = max_probs.ge(u_thres).float()  # 置信度高于阈值的样本保留
        top_br_target_unlabel = max_idx  # 伪标签
        bottom_br_target_unlabel = torch.ge(bottom_br_pred_unlabel, 0.7).float()  # 底部分支二值化

        # 无标签损失计算（使用伪标签）
        loss_unlabel = spot_loss(top_br_target_unlabel, top_br_pred_unlabel,
                                 bottom_br_target_unlabel, bottom_br_pred_unlabel,
                                 mask, mask)  # 无标签损失

        # ------------------------ 对比学习增强 ------------------------
        # 挖掘简单/困难样本对
        easy_dict_unlabel = easy_snippets_mining(top_br_pred_unlabel, feat_unlabel)  # 无标签简单样本
        hard_dict_unlabel = hard_snippets_mining(bottom_br_pred_unlabel, feat_unlabel)  # 无标签困难样本
        easy_dict_label = easy_snippets_mining(top_br_pred, feat)  # 有标签简单样本
        hard_dict_label = hard_snippets_mining(bottom_br_pred, feat)  # 有标签困难样本

        # 计算对比损失
        loss_contrast_unlabel = contrast_loss(easy_dict_unlabel, hard_dict_unlabel)  # 无标签对比损失
        loss_contrast_label = contrast_loss(easy_dict_label, hard_dict_label)  # 有标签对比损失

        # ------------------------ 总损失与优化 ------------------------
        total_loss = loss_label + loss_unlabel + loss_contrast_label + loss_contrast_unlabel  # 总损失

        # 反向传播与参数更新
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch}] 半监督训练总损失: {total_loss.item():.2f}")