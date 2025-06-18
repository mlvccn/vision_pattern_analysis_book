# Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos

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

## 训练

```
python train-msr-small.py
```

