## Requirements

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

使用 P-TAL 方法生成 proposals（不含后处理），或直接使用作者提供的：

```
./data/THUMOS14/*.pkl
```

## 训练&测试

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