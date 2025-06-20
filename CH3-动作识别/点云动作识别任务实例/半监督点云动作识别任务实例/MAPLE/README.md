# 基于掩码伪标记自编码的半监督点云动作识别

该方法来源于国际会议ACM MM2022中“[MAPLE: Masked pseudo-labeling autoEncoder for semi-supervised point cloud action recognition](https://dl.acm.org/doi/pdf/10.1145/3503161.3547892)“一文。

## 环境配置

环境要求：CUDA=11.3，python=3.8，torch=1.10.1

1. 创造虚拟环境

```
conda create --name your_env_name python=3.8
```

2. 激活虚拟环境

```
conda activate --name your_env_name
```

3. 安装依赖

```
pip install -r requirements.txt
```

## 数据准备

1. 下载数据：[Dr Wanqing Li (UOW) - MSR Action3D (google.com)](https://sites.google.com/view/wanqingli/data-sets/msr-action3d)
2. 将压缩包移动到data文件夹下的MSRAction3D中:

    ```bash
    mv Depth.rar data/MSRAction3D/
    ```

3. 解压文件并处理数据:
    ```bash
    cd data/MSRAction3D/
    unrar e Depth.rar
    mkdir point
    python preprocess_file.py --input_dir ./Depth --output_dir ./point
    ```
4. 文件目录如下所示:
    ```text
    MAPLE
    ├── datasets
    ├── modules
    |── data
        │── MSRAction3D
            │-- preprocess_file.py
            │-- Depth
            |-- point
                │-- a01_s01_e01_sdepth.npz
                │-- a01_s01_e02_sdepth.npz
                │-- a01_s01_e03_sdepth.npz
                │-- ...
    
    ```

## 安装PointNet++

```
cd modules
python setup.py install
```

## 训练&测试

 ### Train baseline model

```bash
bash ./train-msr-baseline.sh
```

 ### Training for pseudo label baseline
```bash
bash ./pseudo_labels/train_msr_pseudo.sh
```

 ### Training for MAPLE
```bash
bash ./z_mask/train_mse_msr_gpu0.sh
```

## 在MSRAction3D的实验结果

| Method | Backbone      | 7.5%  | 15.0% | 22.5% | 30.0% | 37.5% |
| ------ | ------------- | ----- | ----- | ----- | ----- | ----- |
| MAPLE  | P4Transformer | 61.95 | 77.10 | 80.47 | 83.16 | 86.53 |

7.5%、15.0%、22.5%、30.0%、37.5%对应掩码比例。

## 致谢

We would like to thank the authors of [MAPLE](https://github.com/SheldongChen/MAPLE/blob/main/README.md) who have significantly accelerated the development of our book.

