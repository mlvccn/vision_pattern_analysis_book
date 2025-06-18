# **Project of  MAPLE**
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
    `── data
        │── MSRAction3D
            │-- preprocess_file.py
            │-- Depth
            `-- point
                │-- a01_s01_e01_sdepth.npz
                │-- a01_s01_e02_sdepth.npz
                │-- a01_s01_e03_sdepth.npz
                │-- ...
    
    ```

## 训练

 ### Train baseline of 5%/10%/20%/30%/40% superveised baseline model

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
