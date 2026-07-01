# 一种结合时间显著性信息重对齐置信度的点级弱监督时序动作定位方法

该方法来源于国际会议CVPR2024的 [Realigning confidence with temporal saliency information for point-level weakly-supervised temporal action localization](https://openaccess.thecvf.com/content/CVPR2024/papers/Xia_Realigning_Confidence_with_Temporal_Saliency_Information_for_Point-Level_Weakly-Supervised_Temporal_CVPR_2024_paper.pdf) 一文。

## 安装


### 要求

- CUDA: 11.3  
- Python: 3.8.16  
- Pytorch: 1.10.0  
- Numpy: 1.20.3  
- Pandas: 1.5.3  
- Tqdm: 4.65.0  

## 数据准备

### 1. 特征抽取

使用由 [LAC](https://github.com/Pilhyeon/Learning-Action-Completeness-from-Points) 提供的 THUMOS14 特征，或从 [GoogleDrive](https://drive.google.com/file/d/1OhHXnGR3nmZf_W9dWlB1d9gIqkMpoAoX/view?usp=drive_link) 下载。

将下载的 `features` 文件夹移动到：

```
./data/THUMOS14/features
```

### 2. 提议生成

使用 P-TAL 方法生成 proposals（不含后处理），或直接使用原作者提供的：

```
./data/THUMOS14/*.pkl
```

## 模型训练&模型测试

### 训练

```bash
bash ./train_thumos.sh 
```

### 测试

1. 从 [GoogleDrive](https://drive.google.com/file/d/1wE89FsNCMb7UwZ3VN16n0M_QHVnZgo-f/view?usp=drive_link) 下载预训练模型  
2. 修改 `./test_thumos.sh` 中的模型路径  
3. 执行测试脚本：

```bash
bash ./test_thumos.sh 
```

## 实验结果

| 模型   | 数据集   | Avg.mAP(0.1:0.7) |
| ------ | -------- | ---------------- |
| TSPNet | THUMOS14 | 57.0             |

## 致谢

我们想要感谢 [TSPNet](https://github.com/zyxia1009/CVPR2024-TSPNet) 的作者们，他们的工作显著加快了我们书籍的开发进程。