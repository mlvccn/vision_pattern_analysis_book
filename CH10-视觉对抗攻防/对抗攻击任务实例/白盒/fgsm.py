	import torch
	import torch.nn.functional as F
	
	def pgd_attack(model, images, labels, eps=8/255, alpha=0.01, iters=20): #L-infinity范数下的PGD攻击
	"""
	Parameters:
	model  : 待攻击模型
	images : 标准化后的干净样本，维度为（批次，高度，宽度，通道数）
	labels : 干净样本对应的真实标记
	eps    : 扰动大小
	alpha  : 攻击步长
	iters  : 攻击迭代次数
	Returns:
	对抗样本
	"""
	images = images.clone().detach().to(torch.float32)
	labels = labels.clone().detach()
	ori_images = images.clone().detach()
	delta = torch.zeros_like(images).uniform_(-eps, eps).to(images.device) #初始化对抗扰动，与干净样本尺寸一致
	delta.requires_grad = True 
	for _ in range(iters):
	outputs = model(images + delta) #将对抗样本输入待攻击模型
	loss = F.cross_entropy(outputs, labels) #求对抗样本的损失函数
	loss.backward()
	grad = delta.grad.detach() #获得对抗样本的梯度
	delta.data = delta + alpha * torch.sign(grad) #更新对抗扰动
	delta.data = torch.clamp(delta, -eps, eps) #将对抗扰动截断在一定范围内
	delta.data = torch.clamp(ori_images + delta, 0, 1) - ori_images #防止对抗样本超过[0,1]边界
	delta.grad.zero_() #清空对抗扰动的梯度，准备下一次迭代
	adv_images = ori_images + delta.detach()
	return adv_images    #返回对抗样本		