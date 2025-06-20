# 代码实践题

在一个大型物流仓库中，工人们需要不断地进行搬运、打包、上架等作业。为了提高工作效率并确保安全，仓库安装多个 3D 激光雷达（LiDAR）传感器，以捕获实时的三维点云数据，反映工人的位置、姿势及动作轨迹。现在已经使用P4Transformer技术设计好一个识别系统，可以自动识别搬运动作（弯腰、抬举）以监测工人的姿势是否符合安全规范，及时提醒错误姿势，减少腰背受伤风险；还可以识别走动、转身等动作，追踪工人的路线，优化仓库作业流线；同时实时监测危险动作（滑倒、跌倒）并立即报警，防止事故扩大。当工人需要弯腰搬运货物时，系统通过点云序列中的姿态变化自动检测腰部弯曲角度和抬举姿势是否安全，如检测到危险动作，立刻发出警报并通知现场管理人员。但是最近需要延长检测时间，旧的系统出现一些问题。经过相关专家分析，这有可能是因为点云视频在时间维度上是有序的，在空间上是无序的，所以当检测时间较长时，对应的点云视频长度增大，P4Transformer在特征提取时容易出现时空紊乱。借鉴相关领域方法，专家认为将时空解耦可以缓解这一现象，现在请参考PSTNet中的时空解耦方式修改P4Transformer中的局部区域特征提取模块的的代码，使其实现时间和空间上的解耦，并使用MSRAction3D数据集验证方法是否有效。

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

## 在MSRAction3D的实验结果

| Method        | Frames | Accuracy |
| ------------- | ------ | -------- |
| P4Transformer | 24     | 90.59    |

## 致谢

我们想要感谢[P4Transformer](https://github.com/hehefan/P4Transformer) and [PSTNet](https://github.com/hehefan/PSTNet2)的作者，他们的工作显著加快了我们书籍的开发进程。
