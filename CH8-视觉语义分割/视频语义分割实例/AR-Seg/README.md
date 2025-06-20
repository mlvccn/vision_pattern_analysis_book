# AR-Seg

本仓库为 AR-Seg 的 PyTorch 实现，适用于压缩视频的高效语义分割。包含 CamVid 和 Cityscapes 数据集的训练、评估与推理脚本。

---

## 环境配置

### 使用 Conda 配置环境

```bash
conda env create -f environment.yml
conda activate AR-Seg
```

### 手动分步安装

```bash
conda create -n AR-Seg python=3.6 
conda activate AR-Seg

conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

---

## 数据集与预处理

请参考 [pre-process/README.md](./pre-process/README.md) 进行数据集准备和预处理。

---

## 预训练模型与示例数据

- 训练权重可从 [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/bb4bedf8c7af4ec8a5b2/) 或 [GoogleDrive](https://drive.google.com/file/d/1u3CUNoRRDi1V1Y4b5Hv8gFwYGdKjXJtp/view?usp=share_link) 下载，解压至 `./checkpoints/` 目录。
- 示例处理后数据可从 [TsinghuaCloud](https://cloud.tsinghua.edu.cn/d/f358201e9ac14c4e801a/) 或 [GoogleDrive](https://drive.google.com/drive/folders/1EMDyP59-WE2OK8FqYFY_f3SAd7PgZ7Ld?usp=share_link) 下载，解压至 `./data/` 目录。

---

## 评估

运行评估脚本：

```bash
python evaluation.py --dataset [camvid 或 cityscapes] --backbone [psp18 或 bise18] --mode [1 1 1 或 0 0 1 等]
```

示例：评估 CamVid 上 BiseNet-18 的 HR 分支：

```bash
python evaluation.py --dataset camvid --backbone bise18 --mode 1 0 0 
```

评估结果将保存在 `./evaluation-result` 目录下。

---

## 训练

### 数据集软链接

CamVid:

```bash
ln -s camvid_root ./data/CamVid
ln -s camvid_sequence_root ./data/camvid-sequence
```

Cityscapes:

```bash
ln -s cityscapes_root ./data/cityscapes
ln -s cityscapes_root/leftImg8bit_sequence ./data/cityscapes-sequence
```

### HR 分支训练

CamVid:

```bash
# PSPNet-18
python train.py --data-path=./data/CamVid/ --models-path=./exp/pspnet18-camvid/scale1.0_epoch100_pure --backend='resnet18' --batch-size=8 --epochs=100 --scale=1.0 --gpu=4

# BiseNet-18
python train.py --data-path=./data/CamVid/ --models-path=./exp/bisenet18-camvid/scale1.0_epoch100_pure --backend='resnet18' --batch-size=8 --epochs=100 --scale=1.0 --gpu=7 --model_type=bisenet
```

Cityscapes:

```bash
# PSPNet-18
python train.py --data-path=./data/cityscapes --models-path=./exp/pspnet18-cityscapes/scale1.0_epoch200_pure_bs8_0.5-2.0-aug-512x1024-lr-0.01-semsegPSP --backend='resnet18' --batch-size=8 --epochs=200 --scale=1.0 --gpu=4 --start-lr=0.01 --model_type=pspnet --dataset=cityscapes
```

### LR 分支训练

CamVid:

```bash
# PSPNet-18
python train_pair.py --data-path=./data/CamVid/ --sequence-path=./data/camvid-sequence --models-path=./exp/pspnet18-camvid/paper/camvid-psp18-scale0.5-3M-GOP12-30fps/ --backend='resnet18' --batch-size=8 --epochs=100 --scale=0.5 --gpu=0,1 --feat_loss=mse  --stage1_epoch=50 --ref_gap=12 --with_motion=1

# BiseNet-18
python train_pair.py --data-path=./data/CamVid/ --sequence-path=./data/camvid-sequence --models-path=./exp/bisenet18-camvid/paper/camvid-bise18-scale0.5-3M-GOP12-30fps/ --backend='resnet18' --batch-size=8 --epochs=100 --scale=0.5 --gpu=0 --feat_loss=mse  --stage1_epoch=50 --ref_gap=12 --with_motion=1 --model_type=bisenet
```

Cityscapes:

```bash
# PSPNet-18
python convert_model_for_cityscapes.py --backbone psp18

python train_pair.py --data-path=./data/cityscapes --sequence-path=./data/cityscapes-sequence --models-path=./exp/pspnet18-cityscapes/paper/cityscapes-psp18-scale0.5-5M-GOP12-30fps_0.01_epoch200-semseg-auxLoss/ --backend='resnet18' --batch-size=8 --epochs=200 --scale=0.5 --gpu=1,2 --feat_loss=mse  --stage1_epoch=0 --ref_gap=12 --with_motion=1 --model_type=pspnet --start-lr=0.01 --dataset=cityscapes --bitrate=5

# BiseNet-18
python convert_model_for_cityscapes.py --backbone bise18

python train_pair.py --data-path=./data/cityscapes --sequence-path=./data/cityscapes-sequence --models-path=./exp/bisenet18-cityscapes/paper/cityscapes-bise18-scale0.5-5M-GOP12-30fps_0.01_epoch200 --backend='resnet18' --batch-size=16 --epochs=200 --scale=0.5 --gpu=2 --feat_loss=mse  --start-lr=0.01 --stage1_epoch=0 --ref_gap=12 --with_motion=1 --model_type=bisenet --dataset=cityscapes --bitrate=5
```

如需在 Cityscapes 上训练 BiseNet，请下载初始化权重并解压至 `./cityscapes_pretrained/` 目录。

---

## 致谢

We would like to thank the authors of [AR-Seg
](https://github.com/THU-LYJ-Lab/AR-Seg) which has significantly accelerated the development of our book.

