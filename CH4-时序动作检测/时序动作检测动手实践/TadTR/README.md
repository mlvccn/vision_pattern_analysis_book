# TadTR: End-to-end Temporal Action Detection with Transformer

本仓库包含论文 [End-to-end temporal action detection with Transformer](https://arxiv.org/abs/2106.10271) 中描述的 TadTR 项目代码，该论文发表于 IEEE Transactions on Image Processing (TIP) 2022。

## 安装

### 要求

* Linux 或 Windows
  
* Python>=3.7

* (可选) CUDA>=9.2, GCC>=5.4
  
* PyTorch>=1.5.1, torchvision>=0.6.1（按照[此处](https://pytorch.org/)的说明安装）
  
* 其他依赖项
    ```bash
    pip install -r requirements.txt
    ```

### 编译 CUDA 扩展（可选）

RoIAlign 运算符使用了 CUDA 扩展实现。如果你的机器有 NVIDIA GPU 并支持 CUDA，可以运行此步骤。否则，请在 `opts.py` 中设置 `disable_cuda=True`。
```bash
cd model/ops;

# 如果你有多版本 CUDA 工具包安装，建议添加前缀
# CUDA_HOME=<你的CUDA工具包路径> 来指定正确的版本。 
python setup.py build_ext --inplace
```

## 数据准备

目前本论文只支持 `thumos14` 数据集。

### THUMOS14

从 [[百度网盘(提取码: adTR)]](https://pan.baidu.com/s/183VprlbKNjMb3Gr-rfmROQ) 或 [[OneDrive]](https://husteducn-my.sharepoint.com/:f:/g/personal/liuxl_hust_edu_cn/EsMyXDlkrTdBsikoRQSIeUsBkxJJRsplbMyIQVYotiZRIQ?e=QYgiCH) 下载所有数据。

- **特征**：下载 I3D 特征 `I3D_2stream_Pth.tar`。这是由 P-GCN 的作者提供的。我已经将 RGB 和 Flow 特征进行了拼接（如果长度不一致则截断较长的一个），并将数据转换为 float32 精度以节省空间。
- **注释**：动作实例的注释和特征文件的元信息。两者都是 JSON 格式 (`th14_annotations_with_fps_duration.json` 和 `th14_i3d2s_ft_info.json`)。
- **预训练参考模型**：预训练模型使用了 I3D 特征 `thumos14_i3d2s_tadtr_reference.pth`。此模型对应配置文件 `configs/thumos14_i3d2s_tadtr.yml`。

下载完成后，通过以下命令解压归档的特征文件：
```bash
cd data
tar -xf I3D_2stream_Pth.tar
```

然后将特征、注释和模型放在 `data/thumos14` 目录下。期望根目录下的结构如下：

```
- data
  - thumos14
    - I3D_2stream_Pth
     - xxxxx
     - xxxxx
    - th14_annotations_with_fps_duration.json
    - th14_i3d2s_ft_info.json
    - thumos14_tadtr_reference.pth
```

## 测试预训练模型

运行以下命令：

```
python main.py --cfg CFG_PATH --eval --resume CKPT_PATH
```

其中 `CFG_PATH` 是定义实验设置的 YAML 格式配置文件的路径，例如 `configs/thumos14_i3d2s_tadtr.yml`。`CKPT_PATH` 是预训练模型的路径。或者，你可以执行 Shell 脚本：

```
bash scripts/test_reference_models.sh thumos14
```

## 自行训练

运行以下命令：

```
python main.py --cfg CFG_PATH
```

这个代码库支持在 CPU 和 GPU 上运行：

- **在 CPU 上运行**：请在上述命令中添加 `--device cpu`。还需要在 `opts.py` 中设置 `disable_cuda=True`。CPU 模式不支持 actionness 回归，检测性能较低。
- **在 GPU 上运行**：由于模型非常轻量，只需一个 GPU 就足够了。你可以通过在上述命令前添加前缀 `CUDA_VISIBLE_DEVICES=ID` 来指定使用的 GPU 设备 ID（例如 0）。要在多个 GPU 上运行，请参考 `scripts/run_parallel.sh`。

在训练期间，我们的代码会每隔 N 个 epoch 自动进行测试（N 是 `opts.py` 中的 `test_interval`）。如果你使用现代 GPU（如 TITAN Xp），THUMOS14 的训练时间约为 6~10 分钟。你还可以使用 Tensorboard 监控训练过程（需要在 `opts.py` 中设置 `cfg.tensorboard=True`）。Tensorboard 记录和检查点将保存在 `output_dir`（可以在配置文件中修改）。

训练完成后，你也可以通过运行以下命令来测试你训练的模型：

```
python main.py --cfg CFG_PATH --eval
```

它会自动使用最佳模型检查点。如果你想手动指定模型检查点，运行：

```
python main.py --cfg CFG_PATH --eval --resume CKPT_PATH
```

