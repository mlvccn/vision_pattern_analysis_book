# Knowledge Transfer with Simulated Inter-Image Erasing for Weakly Supervised Semantic Segmentation

本仓库为 KTSE 的 PyTorch 实现，适用于弱监督语义分割任务，支持 PASCAL VOC 2012 数据集。

---

## 网络结构

![network](framework.png)

---

## 环境配置

- 推荐环境：Ubuntu 18.04，Python 3.8，PyTorch 1.8.2，CUDA 11.3
- 可通过如下命令创建环境：
```bash
conda env create -f wsss_new.yaml
```

---

## 数据集准备

1. 下载 [PASCAL VOC 2012 development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)，放置于 `./data/` 目录下。

---

## 训练与测试

### 1. KTSE 第一阶段训练与测试

- 下载 ImageNet 预训练权重 [ilsvrc-cls_rna-a1_cls1000_ep-0001.params](https://ktse.oss-cn-shanghai.aliyuncs.com/ilsvrc-cls_rna-a1_cls1000_ep-0001.params)，放入 `./pretrained/` 目录。
- 训练 KTSE：
```bash
python train.py --name ktse1 --model ktse
```
- 下载预训练模型 [039net_main.pth](https://ktse.oss-cn-shanghai.aliyuncs.com/039net_main.pth)（PASCAL, seed: 67% mIoU），放入 `./experiments/ktse1/ckpt/` 目录。
- 推理与评估：
```bash
python infer.py --name ktse1 --model ktse --load_epo 39 --dict  --infer_list voc12/train_aug.txt
python evaluation.py --name ktse1 --task cam --dict_dir dict
```

---

### 2. BECO 分割网络训练与测试

- 安装 Python 3.8、PyTorch 1.11.0 及 requirements.txt 依赖。
- 下载 DeeplabV2 的 ImageNet 预训练模型 [resnet101-cd907fc2.pth](https://download.pytorch.org/models/resnet101-cd907fc2.pth)，重命名为 `resnet-101_v2.pth`，放入 `./data/model_zoo/`。
- 下载伪标签 [sem_seg](https://ktse.oss-cn-shanghai.aliyuncs.com/sem_seg.zip)，解压至 `./data/`。
- 下载分割网络预训练权重 [best_ckpt_KTSE_73.0.pth](https://ktse.oss-cn-shanghai.aliyuncs.com/best_ckpt_KTSE_73.0.pth)，放入 `./segmentation/` 目录。

- 测试分割网络（如需 CRF 后处理，需安装 pydensecrf）：
```bash
cd segmentation
pip install -r requirements.txt 
python main.py --test --logging_tag seg_result --ckpt best_ckpt_KTSE_73.0.pth
python test.py --crf --logits_dir ./data/logging/seg_result/logits_msc --mode "val"
```

---

### 3. IRN 伪标签细化

- 下载 [039net_main.pth](https://ktse.oss-cn-shanghai.aliyuncs.com/039net_main.pth) 并放入 `./irn/sess/` 目录。
- 生成伪标签和置信度掩码（可直接下载 [sem_seg](https://ktse.oss-cn-shanghai.aliyuncs.com/sem_seg.zip) 和 [mask_irn](https://ktse.oss-cn-shanghai.aliyuncs.com/mask_irn.zip)）：
```bash
cd irn 
python run_sample.py
python gen_mask.py
```

---

### 4. 分割网络训练

- 数据与预训练模型目录结构示例：
```
data/
    ├── VOC2012/
    ├── sem_seg/
    ├── mask_irn/
    ├── model_zoo/
    └── logging/
```
- 训练分割网络：
```bash
cd segmentation
python main.py -dist --logging_tag seg_result --amp
```

---

## 致谢

We would like to thank the authors of [KTSE](https://github.com/chentao2016/KTSE)] which has significantly accelerated the development of our book.



