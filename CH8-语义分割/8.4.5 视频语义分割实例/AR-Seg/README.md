# 压缩视频的高效语义分割分辨率自适应方法

This repository provides the PyTorch implementation of AR-Seg for efficient semantic segmentation on compressed videos. It includes training, evaluation, and inference scripts for the CamVid and Cityscapes datasets.

---

## Environment Setup

### Using Conda Environment

```bash
conda env create -f environment.yml
conda activate AR-Seg
```

### Manual Installation

```bash
conda create -n AR-Seg python=3.6 
conda activate AR-Seg

conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

---

## Dataset and Preprocessing

Please refer to [pre-process/README.md](./pre-process/README.md) for dataset preparation and preprocessing instructions.

---

## Pretrained Models and Example Data

- Training weights can be downloaded from [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/bb4bedf8c7af4ec8a5b2/) or [GoogleDrive](https://drive.google.com/file/d/1u3CUNoRRDi1V1Y4b5Hv8gFwYGdKjXJtp/view?usp=share_link) and extracted to the `./checkpoints/` directory.
- Example processed data can be downloaded from [TsinghuaCloud](https://cloud.tsinghua.edu.cn/d/f358201e9ac14c4e801a/) or [GoogleDrive](https://drive.google.com/drive/folders/1EMDyP59-WE2OK8FqYFY_f3SAd7PgZ7Ld?usp=share_link) and extracted to the `./data/` directory.

---

## Evaluation

Run the evaluation script:

```bash
python evaluation.py --dataset [camvid or cityscapes] --backbone [psp18 or bise18] --mode [1 1 1 or 0 0 1 etc.]
```

Example: Evaluate the HR branch of BiseNet-18 on CamVid:

```bash
python evaluation.py --dataset camvid --backbone bise18 --mode 1 0 0 
```

Evaluation results will be saved in the `./evaluation-result` directory.

---

## Training

### Dataset Symlink

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

### HR Branch Training

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

### LR Branch Training

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

If you want to train BiseNet on Cityscapes, please download the initialization weights and extract them to the `./cityscapes_pretrained/` directory.

---

## Acknowledgement

We would like to thank the authors of [AR-Seg](https://github.com/THU-LYJ-Lab/AR-Seg) which has significantly accelerated the development of our book.

