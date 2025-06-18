## 环境依赖

- PyTorch 1.7.1  
- pytorch-lightning 1.1.2  
- 还需安装 loguru、colorama 等依赖  

推荐使用以下命令创建 conda 环境（注意，请适配你自己的CUDA版本安装对应的pytorch等相关包，否则会报错）：

```bash
conda env create -f environment.yml
```

## 运行步骤

### 数据下载

从以下链接下载 THUMOS14 和 ActivityNet1.2 的 ground-truth 和 I3D 特征：

[Box 数据链接](https://rpi.box.com/s/hf6djlgs7vnl7a2oamjt0vkrig42pwho)

下载完成后，将文件/文件夹放置在：

```
./data/
```

### 模型训练（THUMOS14）

执行以下命令开始训练：

```bash
python main.py
```

可通过 `options.py` 查看更多命令行参数选项。

### 模型测试（THUMOS14）

执行以下命令进行测试：

```bash
python main.py --test --ckpt [checkpoint_path]
```

将 `[checkpoint_path]` 替换为你的模型路径。

### 训练/测试 ActivityNet-1.2

使用下列脚本运行：

```bash
python main_anet.py
```
