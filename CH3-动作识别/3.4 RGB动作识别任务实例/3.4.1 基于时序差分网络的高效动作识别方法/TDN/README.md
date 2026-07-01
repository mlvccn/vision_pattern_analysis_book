# 基于时序差分网络的高效动作识别方法

该方法来源于国际会议CVPR 2021中“[TDN: Temporal Difference Networks for Efficient Action Recognition](https://arxiv.org/abs/2012.10071)“一文

## 环境配置

- Python 3.6 or higher
- [PyTorch](https://pytorch.org/) **1.4** or higher
- [Torchvision](https://github.com/pytorch/vision)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm.git)
- [scikit-learn](https://scikit-learn.org/stable/)
- [ffmpeg](https://www.ffmpeg.org/)  
- [decord](https://github.com/dmlc/decord) 

## 数据准备
下载数据：

 [Kinetics400](https://deepmind.com/research/open-source/kinetics), [UCF101](https://www.crcv.ucf.edu/data/UCF101.php), [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/), [Something-Something-V1](https://20bn.com/datasets/something-something/v1)

- **Something-Something-V1 和 V2** 的处理可以总结为三个步骤
    
    1. 从视频中提取帧（可以使用 ffmpeg 从视频中获取帧）      
    2. 生成数据加载器所需的标注文件（标注中格式为“<path_to_frames> <frames_num> <video_class>”）。通常标注文件包含 train.txt 和 val.txt。*.txt 文件的格式如下：
        ```
        dataset_root/frames/video_1 num_frames label_1
        dataset_root/frames/video_2 num_frames label_2
        dataset_root/frames/video_3 num_frames label_3
        ...
        dataset_root/frames/video_N num_frames label_N
        ```
    3. 将信息添加到 `ops/dataset_configs.py` 中。
    
- Kinetics400的处理过程可概括为三个步骤：
    1. 我们通过将视频的短边调整为320像素来预处理数据。您可以参考[MMAction2数据基准](https://github.com/open-mmlab/mmaction2)中的[TSN](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn#kinetics-400-data-benchmark-8-gpus-resnet50-imagenet-pretrain-3-segments)和[SlowOnly](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly#kinetics-400-data-benchmark)。
    2. 生成数据加载器所需的标注文件（标注中格式为“<视频路径> <视频类别>”）。标注通常包含 train.txt 和 val.txt 文件。*.txt 文件的格式如下：
        ```
        dataset_root/video_1.mp4  label_1
        dataset_root/video_2.mp4  label_2
        dataset_root/video_3.mp4  label_3
        ...
        dataset_root/video_N.mp4  label_N
        ```
    3. 将信息添加到 `ops/dataset_configs.py` 中。


## Testing

1. 运行以下测试脚本：
    ```
    CUDA_VISIBLE_DEVICES=0 python3 test_models_center_crop.py something \
    --archs='resnet50' --weights <your_checkpoint_path>  --test_segments=8  \
    --test_crops=1 --batch_size=16  --gpus 0 --output_dir <your_pkl_path> -j 4 --clip_index=0
    ```
2. 运行以下脚本以获取原始分数的结果：
    ```
    python3 pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir <your_pkl_path>  
    ```

## Training

其实现支持多GPU和分布式数据并行训练，速度更快且更简单。
- 例如，使用 8 个 GPU 训练 TDN-ResNet50 在 Something-Something-V1 上时，可以运行：
    ```
    python -m torch.distributed.launch --master_port 12347 --nproc_per_node=8 \
                main.py  something  RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.01 \
                --lr_scheduler step --lr_steps  30 45 55 --epochs 60 --batch-size 8 \
                --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb 
    ```
- 例如，要在 Something-Something-V1 上训练 TDN-ResNet50，使用 8 个 GPU，可以运行：
    ```
    python -m torch.distributed.launch --master_port 12347 --nproc_per_node=8 \
            main.py  kinetics RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.02 \
            --lr_scheduler step  --lr_steps 50 75 90 --epochs 100 --batch-size 16 \
            --wd 1e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb 
    ```

## 致谢

我们想要感谢[TDN](https://github.com/MCG-NJU/TDN)的作者，他们的工作显著加快了我们书籍的开发进程。

