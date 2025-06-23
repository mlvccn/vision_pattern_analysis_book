# 对比提议学习（Contrastive Proposal Learning, CPL）

该典型方法来源于国际会议CVPR2022"[Weakly Supervised Temporal Sentence Grounding with Gaussian-based Contrastive Proposal Learning](https://minghangz.github.io/uploads/CPL/CPL_paper.pdf)"一文。

## 1.所需环境

```
- pytorch 1.17.1
- h5py 3.11.0
- nltk 3.9.1
- fairseq 0.10.2
```

## 2.数据准备

请准备以下特征文件并按如下结构放置于 `data/` 目录下：

```
data
├── activitynet
│ ├── sub_activitynet_v1-3.c3d.hdf5 # C3D特征（下载地址：http://activity-net.org/challenges/2016/download.html）
│ ├── glove.pkl
│ ├── train_data.json
│ ├── val_data.json
│ ├── test_data.json
├── charades
│ ├── i3d_features.hdf5 # 下载地址：https://pan.baidu.com/s/1WhWreaHIx8pI5hLK2uyCdw?pwd=4g9h
│ ├── glove.pkl
│ ├── train.json
│ ├── test.json
```

注：Charades-STA 的 I3D 特征来自 LGI 项目（https://github.com/JonghwanMun/LGI4temporalgrounding），若为 `.npy` 格式需用提供的 `convert_npy_to_hdf5.py` 脚本转换为 `.hdf5` 格式。

## 3.训练命令

### 在 ActivityNet Captions 数据集上训练：

```bash
python train.py --config-path config/activitynet/main.json --log_dir LOG_DIR --tag TAG
```

###  Charades-STA 数据集上训练：

```bash
python train.py --config-path config/charades/main.json --log_dir LOG_DIR --tag TAG
```

说明：

- `--log_dir`：日志保存目录（可选）
- `--tag`：实验标签（可选）
- 权重默认保存在 `checkpoints/` 文件夹中，可在配置文件中修改路径。

## 4.推理命令（测试）

训练好的模型保存在 `checkpoints/` 文件夹下，使用以下命令进行评估：

```bash
# 使用 loss-based 策略推理
python train.py --config-path CONFIG_FILE --resume CHECKPOINT_FILE --eval

# 使用 vote-based 策略推理
python train.py --config-path CONFIG_FILE --resume CHECKPOINT_FILE --eval --vote
```

说明：

- `CONFIG_FILE` 与训练时使用的配置文件一致
- `CHECKPOINT_FILE` 为训练保存的模型路径（如：`checkpoints/charades/model-best.pt`）

## 5.实验结果

### Charades-STA Dataset

| 推理策略 | R1@0.3 | R1@0.5 | R1@0.7 | R5@0.3 | R5@0.5 | R5@0.7 |
| :------: | :----: | :----: | :----: | :----: | :----: | :----: |
| 基于损失 | 66.40  | 49.24  | 22.39  | 96.99  | 84.71  | 52.37  |
| 基于投票 | 65.99  | 49.05  | 22.61  | 96.99  | 84.71  | 52.37  |

### ActivityNet Captions Dataset

| 推理策略 | R1@0.3 | R1@0.5 | R1@0.7 | R5@0.3 | R5@0.5 | R5@0.7 |
| :------: | :----: | :----: | :----: | :----: | :----: | :----: |
| 基于损失 | 79.86  | 53.67  | 31.24  | 87.24  | 63.05  | 43.13  |
| 基于投票 | 82.55  | 55.73  | 31.37  | 87.24  | 63.05  | 43.13  |

## 致谢

我们想要感谢[CPL](https://github.com/minghangz/cpl)的作者们，他们的工作显著加快了我们书籍的开发进程。
