# DeepLabv3Plus-Pytorch

本仓库为 DeepLabv3/DeepLabv3+ 的 PyTorch 实现，支持 Pascal VOC 2012、Cityscapes 等数据集。

---

## 环境配置

推荐环境：Python 3.8+，PyTorch 1.8+，CUDA 11.1+

```bash
conda create -n deeplabv3plus python=3.8 -y
conda activate deeplabv3plus
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

---

## 支持的网络结构

| DeepLabV3                | DeepLabV3+                |
| :----------------------: | :-----------------------: |
| deeplabv3_resnet50       | deeplabv3plus_resnet50    |
| deeplabv3_resnet101      | deeplabv3plus_resnet101   |
| deeplabv3_mobilenet      | deeplabv3plus_mobilenet   |
| deeplabv3_hrnetv2_48     | deeplabv3plus_hrnetv2_48  |
| deeplabv3_hrnetv2_32     | deeplabv3plus_hrnetv2_32  |
| deeplabv3_xception       | deeplabv3plus_xception    |

更多模型请参考 [network/modeling.py](network/modeling.py)。

---

## 预训练模型与权重

- [Dropbox 下载](https://www.dropbox.com/sh/w3z9z8lqpi8b2w7/AAB0vkl4F5vy6HdIhmRCTKHSa?dl=0)
- [腾讯微云下载](https://share.weiyun.com/qqx78Pv5)
- HRNet 预训练权重：[Google Drive](https://drive.google.com/file/d/1NxCK7Zgn5PmeS7W1jYLt5J9E0RRZ2oyF/view?usp=sharing)

下载后放入 `checkpoints/` 目录。

---

## 数据集准备

### Pascal VOC

1. 标准 Pascal VOC：可通过 `--download` 选项自动下载，默认路径为 `./datasets/data`。
2. 推荐使用 trainaug 增强集，需下载 [SegmentationClassAug 标签](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) 并解压到 `VOC2012` 目录。

目录结构示例：

```
/datasets
    /data
        /VOCdevkit  
            /VOC2012
                /SegmentationClass
                /SegmentationClassAug
                /JPEGImages
```

### Cityscapes

下载并解压到 `datasets/data/cityscapes`：

```
/datasets
    /data
        /cityscapes
            /gtFine
            /leftImg8bit
```

---

## 训练与测试

### Pascal VOC 训练示例

```bash
# 启动 visdom 可视化（可选）
visdom -port 28333

# 训练（以 MobileNet 为例）
python main.py --model deeplabv3plus_mobilenet --enable_vis --vis_port 28333 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16
```

### Pascal VOC 测试示例

```bash
python main.py --model deeplabv3plus_mobilenet --enable_vis --vis_port 28333 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --ckpt checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth --test_only --save_val_results
```

### Cityscapes 训练示例

```bash
python main.py --model deeplabv3plus_mobilenet --dataset cityscapes --enable_vis --vis_port 28333 --gpu_id 0 --lr 0.1 --crop_size 768 --batch_size 16 --output_stride 16 --data_root ./datasets/data/cityscapes 
```

### 单张图片或文件夹预测

```bash
python predict.py --input [图片路径或文件夹] --dataset cityscapes --model deeplabv3plus_mobilenet --ckpt [权重路径] --save_val_results_to [输出目录]
```

---

## 模型性能

### Pascal VOC2012 Aug (21类, 513x513)

|  Model                    | Batch Size | FLOPs  | train/val OS |  mIoU   | 权重下载 |
| :-----------------------: | :--------: | :----: | :----------: | :-----: | :------: |
| DeepLabV3-MobileNet       | 16         | 6.0G   | 16/16        | 0.701   | [下载](https://www.dropbox.com/s/uhksxwfcim3nkpo/best_deeplabv3_mobilenet_voc_os16.pth?dl=0) |
| DeepLabV3-ResNet50        | 16         | 51.4G  | 16/16        | 0.769   | [下载](https://www.dropbox.com/s/3eag5ojccwiexkq/best_deeplabv3_resnet50_voc_os16.pth?dl=0) |
| DeepLabV3-ResNet101       | 16         | 72.1G  | 16/16        | 0.773   | [下载](https://www.dropbox.com/s/vtenndnsrnh4068/best_deeplabv3_resnet101_voc_os16.pth?dl=0) |
| DeepLabV3Plus-MobileNet   | 16         | 17.0G  | 16/16        | 0.711   | [下载](https://www.dropbox.com/s/0idrhwz6opaj7q4/best_deeplabv3plus_mobilenet_voc_os16.pth?dl=0) |
| DeepLabV3Plus-ResNet50    | 16         | 62.7G  | 16/16        | 0.772   | [下载](https://www.dropbox.com/s/dgxyd3jkyz24voa/best_deeplabv3plus_resnet50_voc_os16.pth?dl=0) |
| DeepLabV3Plus-ResNet101   | 16         | 83.4G  | 16/16        | 0.783   | [下载](https://www.dropbox.com/s/bm3hxe7wmakaqc5/best_deeplabv3plus_resnet101_voc_os16.pth?dl=0) |

### Cityscapes (19类, 1024x2048)

|  Model                    | Batch Size | FLOPs  | train/val OS |  mIoU   | 权重下载 |
| :-----------------------: | :--------: | :----: | :----------: | :-----: | :------: |
| DeepLabV3Plus-MobileNet   | 16         | 135G   | 16/16        | 0.721   | [下载](https://www.dropbox.com/s/753ojyvsh3vdjol/best_deeplabv3plus_mobilenet_cityscapes_os16.pth?dl=0) |
| DeepLabV3Plus-ResNet101   | 16         |  N/A   | 16/16        | 0.762   | [下载](https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb/view?usp=sharing) |

---

## 许可

本仓库代码仅供学术研究使用，禁止商业用途。

---

## 致谢

We would like to thank the authors of [DeepLabv3Plus](https://github.com/VainF/DeepLabV3Plus-Pytorch)which has significantly accelerated the development of our book.
