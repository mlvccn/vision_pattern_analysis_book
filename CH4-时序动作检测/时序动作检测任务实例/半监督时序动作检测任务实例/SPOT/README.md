## 开始使用

### 系统需求
- Python 3.7
- PyTorch == 1.9.0  **(请确保您的PyTorch版本至少为1.8)**
- NVIDIA GPU
- Kornia库

### 环境设置
建议创建一个Conda环境并安装以下依赖：
```shell
pip3 install -r requirements.txt
```

### 下载特征
下载视频特征，并在 `config/anet.yaml` 文件中更新视频路径/输出路径。目前仅提供了ActivityNetv1.3数据集的配置文件。我们计划很快发布对THUMOS14数据集的支持。

| 数据集       | 特征主干        | 预训练      | 链接                                                                 |
|:------------:|:---------------:|:-----------:|:--------------------------------------------------------------------:|
| ActivityNet  | TSN             | Kinetics-400 | [Google Drive](https://drive.google.com/u/0/uc?id=1ISemndlSDS2FtqQOKL0t3Cjj9yk2yznF&export=download) |
| THUMOS       | TSN             | Kinetics-400 | [Google Drive](https://drive.google.com/drive/folders/1-19PgCRTTNfy2RWGErvUUlT0_3J-qEb8?usp=sharing) |
| ActivityNet  | I3D             | Kinetics-400 | [Google Drive(文件找不到了)](https://drive.google.com/drive/folders/1B1srfie2UWKwaC4-7bo6UItmJoESCUq3?usp=sharing) |
| THUMOS       | I3D             | Kinetics-400 | [Google Drive](https://drive.google.com/drive/folders/1C4YG01X9IIT1a568wMM8fgm4k4xTC2EQ?usp=sharing) |

### 模型训练

要从头开始训练SPOT，请运行以下命令。您可以通过修改 `config/anet.yaml` 文件来调整训练配置。这包括预训练和微调阶段。

```
python spot_train.py
```

### 模型推理
原作者提供了在ActivityNetv1.3上使用I3D特征的预训练模型，可在此[链接](https://drive.google.com/file/d/1ltF5AKee8JcdJmDPabJtXwJe1_m0X3Sc/view?usp=sharing)找到。

下载检查点后，可以在 `config/anet.yaml` 文件中保存检查点路径。然后使用以下命令执行模型推理：
```shell
python spot_inference.py
```
### 模型评估

要评估模型，请运行以下命令：
```
python eval.py
```
