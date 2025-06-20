# 基于点4D Transformer的点云时空建模

该方法来源于国际会议CVPR2021中“[Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Point_4D_Transformer_Networks_for_Spatio-Temporal_Modeling_in_Point_Cloud_CVPR_2021_paper.pdf)“一文。

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

2. 将文件移动到根目录的data文件夹下:

   ```bash
   mv Depth.rar data/
   ```

3. 解压文件并处理数据:

   ```bash
   cd data
   unrar e Depth.rar
   mkdir point
   python ../datasets/preprocess_file.py --input_dir ./Depth --output_dir ./point
   ```

4. 文件目录如下所示:

   ```text
   MAPLE
   ├── datasets
   	│-- preprocess_file.py
   ├── modules
   |── data
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

```
python train-msr-small.py
```

## 在MSRAction3D上的实验结果

|    Method     | Frames | Accuracy |
| :-----------: | :----: | :------: |
| P4Transformer |   24   |  90.94   |

## 致谢

We would like to thank the authors of [P4Transformer](https://github.com/hehefan/P4Transformer) who have significantly accelerated the development of our book.
