# 端到端半监督视频动作检测网络

该方法来源于国际会议CVPR2022的“[End-to-End Semi-Supervised Learning for Video Action Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Kumar_End-to-End_Semi-Supervised_Learning_for_Video_Action_Detection_CVPR_2022_paper.pdf)”一文。

## 数据集准备

1. 下载[JHMDB和UCF101-24数据集](https://pan.baidu.com/s/1D2m05uA6_FeA_ITCeKNgAQ?pwd=4g9r)
2. 将DATASETS文件夹放在pi-consistency-activity-detection-demo目录下
3. cd到pi-consistency-activity-detection-demo目录下，然后解压文件

```bash
cd DATASETS
unzip JHMDB.zip -d JHMDB
unzip UCF101-24.zip -d UCF101-24
```

## 环境安装

cd到pi-consistency-activity-detection-demo目录下

```bash
conda create -n e2essl python=3.9
conda activate e2essl
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install cython==3.0.11
pip install scipy==1.10.1
pip install opencv-python==4.10.0.84
pip install gdown
pip install tensorboard==2.7.0
pip install six==1.16.0
pip install tb_nightly==2.14.0a20230808 -i https://mirrors.aliyun.com/pypi/simple
pip install -r requirements.txt
pip install numpy==1.23 -i https://pypi.mirrors.ustc.edu.cn/simple
pip install --index-url https://mirrors.aliyun.com/pypi/simple/ torchvideo
pip install git+https://github.com/hassony2/torch_videovision 
pip install vidaug
pip install tensorboardx==2.6.2.2
pip install setuptools==59.1.0
pip install scikit-image==0.16.2
pip install protobuf==3.20.0
```

## 下载权重

下载[预训练权重](https://pan.baidu.com/s/1kaESs3oJG1iO_4-undidmA?pwd=xh8b)，并放到“pi-consistency-activity-detection-demo/weights”目录下。

## 运行命令

实验在使用3090显卡上运行

### jhmdb_cyclic_variance_maps

#### 训练指令

```
python main_jhmdb.py --epochs 50 --bs 8 --lr 1e-4 --log_dir result/jhmdb --txt_file_label jhmdb_classes_list_per_30_labeled.txt --txt_file_unlabel jhmdb_classes_list_per_70_unlabeled.txt --wt_cls 1 --wt_cons 0.1 --const_loss l2 --bv --n_frames 5 --thresh_epoch 11 --exp_id cyclic_variance_maps
```

#### 验证指令，计算mAP，注意修改路径

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_jhmdb.py --ckpt /data/hlf/book_code_demo/pi-consistency-activity-detection-demo/result/jhmdb/cyclic_variance_maps/06-04-13-42 --txt_path ./mAP_log/jhmdb_cyclic_variance_maps_results.txt
```

### jhmdb_gradient_maps

#### 训练指令

```
CUDA_VISIBLE_DEVICES=1 python main_jhmdb.py --epochs 50 --bs 8 --lr 1e-4 --log_dir result/jhmdb --txt_file_label jhmdb_classes_list_per_30_labeled.txt --txt_file_unlabel jhmdb_classes_list_per_70_unlabeled.txt --wt_cls 1 --wt_cons 0.1 --const_loss l2 --gv --exp_id gradient_maps
```

#### 验证指令，计算mAP，注意修改路径

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_jhmdb.py --ckpt /data/hlf/book_code_demo/pi-consistency-activity-detection-demo/result/jhmdb/gradient_maps/06-04-13-44 --txt_path ./mAP_log/jhmdb_gradient_maps_results.txt
```

### ucf101-24_cyclic_variance_maps

#### 训练指令

```bash
CUDA_VISIBLE_DEVICES=0 python main_ucf101.py --epochs 100 --bs 8 --loc_loss dice --lr 1e-4 --log_dir result/ucf101-24 --pkl_file_label train_annots_20_labeled_random.pkl --pkl_file_unlabel train_annots_80_unlabeled_random.pkl --wt_loc 1 --wt_cls 1 --wt_cons 0.1 --const_loss l2 --bv --n_frames 5 --thresh_epoch 11 --exp_id cyclic_variance_maps
```

#### 验证指令，计算mAP，注意修改路径

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_ucf101.py --ckpt /data/hlf/book_code_demo/pi-consistency-activity-detection-demo/result/ucf101-24/cyclic_variance_maps/06-04-15-21 --txt_path ./mAP_log/ucf101-24_cyclic_variance_maps_results.txt
```

### ucf101-24_gradient_maps

#### 训练指令

```bash
CUDA_VISIBLE_DEVICES=1 python main_ucf101.py --epochs 100 --bs 8 --loc_loss dice --lr 1e-4 --log_dir result/ucf101-24 --pkl_file_label train_annots_20_labeled_random.pkl --pkl_file_unlabel train_annots_80_unlabeled_random.pkl --wt_loc 1 --wt_cls 1 --wt_cons 0.1 --const_loss l2 --gv --n_frames 5 --thresh_epoch 11 --exp_id gradient_maps
```

#### 验证指令，计算mAP，注意修改路径

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_ucf101.py --ckpt /data/hlf/book_code_demo/pi-consistency-activity-detection-demo/result/ucf101-24/gradient_maps/06-04-15-20 --txt_path ./mAP_log/ucf101-24_gradient_maps_results.txt
```

## 训练命令参数介绍

这是分别用于运行方差和梯度图代码的命令行参数

```
python main.py --epochs 100 --bs 8 --loc_loss dice --lr 1e-4\
 --pkl_file_label train_annots_20_labeled.pkl\
 --pkl_file_unlabel train_annots_80_unlabeled.pkl\
 --wt_loc 1 --wt_cls 1 --wt_cons 0.1\
 --const_loss l2\
 --bv --n_frames 5 --thresh_epoch 11\
 --exp_id cyclic_variance_maps
```

```
python main.py --epochs 100 --bs 8 --loc_loss dice --lr 1e-4\
 --pkl_file_label train_annots_20_labeled.pkl\
 --pkl_file_unlabel train_annots_80_unlabeled.pkl\
 --wt_loc 1 --wt_cls 1 --wt_cons 0.1\
 --const_loss l2\
 --gv\
 --exp_id gradient_maps
```

- 参数说明：

  - *bv* - 时间方差注意力掩码
  - *gv* - 梯度平滑性注意力掩码
  - *wt_loc* - 定位损失的权重
  - *wt_cls* - 分类损失的权重
  - *wt_cons* - 一致性损失的权重
  - *exp_id* - 用于设置保存检查点文件夹名称的实验编号
  - *pkl_file_label* - 已标记子集
  - *pkl_file_unlabel* - 未标记子集

## 实验结果

### UCF101-24

An analysis of the effect of temporal constraints on consistency regularization using
UCF101-24 20% and JHMDB-21 30% labeled subset. V, G, VC and L2 stands for Variance, Gradient,
Cyclic variance and non-attentive L2 loss.

| V    | G            | VC           | L2           | f-mAP@0.5 | v-mAP@0.5 |
| ---- | ------------ | ------------ | ------------ | --------- | --------- |
|      |              | $\checkmark$ | $\checkmark$ | 69.9      | 72.1      |
|      | $\checkmark$ |              | $\checkmark$ | 69.4      | 72.0      |

### JHMDB

An analysis of the effect of temporal constraints on consistency regularization using
UCF101-24 20% and JHMDB-21 30% labeled subset. V, G, VC and L2 stands for Variance, Gradient,
Cyclic variance and non-attentive L2 loss.

| V    | G            | VC           | L2           | f-mAP@0.5 | v-mAP@0.5 |
| ---- | ------------ | ------------ | ------------ | --------- | --------- |
|      |              | $\checkmark$ | $\checkmark$ | 64.4      | 63.5      |
|      | $\checkmark$ |              | $\checkmark$ | 63.1      | 62.2      |

## 致谢

我们想要感谢 [端到端半监督视频动作检测网络](https://github.com/AKASH2907/pi-consistency-activity-detection) 的作者们，他们的工作显著加快了我们书籍的开发进程。
