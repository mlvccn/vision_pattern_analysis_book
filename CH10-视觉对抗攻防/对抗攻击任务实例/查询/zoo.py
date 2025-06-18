	import torch
	import numpy as np
	def zoo_attack(
	model,               # 黑盒模型：输入图像 -> logits
	x,                   # 原始图像 (1, C, H, W)
	y,                   # 正确标签 (1,)
	loss_fn,             # 损失函数，例如 CrossEntropy
	epsilon=0.05,        # 最大扰动范围
	h=1e-4,              # 有限差分步长
	learning_rate=1e-2,  # 更新步长
	query_budget=10000   # 总查询预算
	):
	x = x.clone().detach()
	x_adv = x.clone().detach().requires_grad_(False)
	delta = torch.zeros_like(x_adv)
	query_count = 0
	N = x.numel()
	while(model(x_adv) = y):
	if query_count >= query_budget:
	print("Query budget exhausted.")
	break
	i = np.random.randint(0, N)# 随机选择一个像素方向（坐标索引）
	idx = np.unravel_index(i, x.shape)
	delta_pos = delta.clone()
	delta_neg = delta.clone()
	delta_pos[idx] += h# 构造有限差分方向：+h 和 -h
	delta_neg[idx] -= h
	x_pos = torch.clamp(x_adv + delta_pos, 0, 1)
	x_neg = torch.clamp(x_adv + delta_neg, 0, 1)
	with torch.no_grad():
	loss_pos = loss_fn(model(x_pos), y)# 查询黑盒模型
	loss_neg = loss_fn(model(x_neg), y)
	grad_estimate = (loss_pos - loss_neg) / (2 * h)
	query_count += 2 #更新查询次数
	delta[idx] -= learning_rate * grad_estimate# 梯度下降更新扰动
	delta = torch.clamp(delta, -epsilon, epsilon)
	x_adv = torch.clamp(x + delta, 0, 1)
	return x_adv.detach(), query_count