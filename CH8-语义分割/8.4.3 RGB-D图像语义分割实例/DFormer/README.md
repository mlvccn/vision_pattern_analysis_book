# <p align=center>`DFormer: 用于RGBD语义分割的Transformer`</p>

This repository provides the PyTorch implementation of DFormer and DFormerv2 for RGBD semantic segmentation tasks, supporting mainstream datasets such as NYUDepthv2 and SUNRGBD.

---

## Installation

```bash
conda create -n dformer python=3.10 -y
conda activate dformer

# CUDA 11.8
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

pip install tqdm opencv-python scipy tensorboardX tabulate easydict ftfy regex
```

---

## Datasets and Pretrained Weights

- **Datasets**:  
  By default, place them in the `datasets` folder, or use a soft link `ln -s path_to_data datasets`.

  | Dataset | [GoogleDrive](https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EqActCWQb_pJoHpxvPh4xRgBMApqGAvUjid-XK3wcl08Ug?e=VcIVob) | [BaiduNetdisk](https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q) |
  |:---:|:---:|:---:|:---:|

  The depth `.npy` files have been converted to `.png`, and the paths and split files have been organized.

- **Pretrained Weights**:  
  ImageNet-1K pretrained weights and DFormer/DFormerv2 weights trained on NYUDepth/SUNRGBD can be downloaded below:

  | Weights | DFormer | DFormerv2 |
  |-------|-------| -  |
  | Pretrained | [GoogleDrive](https://drive.google.com/drive/folders/1YuW7qUtnguUFkhC-sfqGySrerjK0rZJX?usp=sharing), [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EhTTF_ZofnFIkz2WSDFAiiIBEIubZUpIwDQYwm9Hvxwu8Q?e=x8XumL), [BaiduNetdisk](https://pan.baidu.com/s/1JlexzFqMcZOXPNiNkE1zRA?pwd=gct6) | [BaiduNetdisk](https://pan.baidu.com/s/1alSvGtGpoW5TRyLxOt1Txw?pwd=i3pn), [HuggingFace](https://huggingface.co/bbynku/DFormerv2/tree/main/DFormerv2/pretrained) |
  | NYUDepthv2 | [GoogleDrive](https://drive.google.com/drive/folders/1P5HwnAvifEI6xiTAx6id24FUCt_i7GH8?usp=sharing), [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/ErAmlYuhS6FCqGQZNGZy0_EBYgJsK3pFTsi2q9g14MEE_A?e=VoKUAf), [BaiduNetdisk](https://pan.baidu.com/s/1AkvlsAvJPv21bz2sXlrADQ?pwd=6vuu) | [BaiduNetdisk](https://pan.baidu.com/s/1hi_XPCv1JDRBjwk8XN7e-A?pwd=3vym), [HuggingFace](https://huggingface.co/bbynku/DFormerv2/tree/main/DFormerv2/NYU) |
  | SUNRGBD | [GoogleDrive](https://drive.google.com/drive/folders/1b005OUO8QXzh0sJM4iykns_UdlbMNZb8?usp=sharing), [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EiNdyUV486BFvb7H2yJWSCMBElOj-m6EppIy4dSXNX-yNw?e=fu2Che), [BaiduNetdisk](https://pan.baidu.com/s/1D6UMiBv6fApV5lafo9J04w?pwd=7ewv) | [BaiduNetdisk](https://pan.baidu.com/s/1NUOgzYmrXmwU7XA8RTRYPg?pwd=ytr7), [HuggingFace](https://huggingface.co/bbynku/DFormerv2/tree/main/DFormerv2/SUNRGBD) |

<details>
<summary>Recommended checkpoints and dataset directory structure:</summary>
<pre><code>
<checkpoints>
|-- <pretrained>
    |-- <DFormer_Large.pth.tar>
    |-- <DFormer_Base.pth.tar>
    |-- <DFormer_Small.pth.tar>
    |-- <DFormer_Tiny.pth.tar>
    |-- <DFormerv2_Large_pretrained.pth>
    |-- <DFormerv2_Base_pretrained.pth>
    |-- <DFormerv2_Small_pretrained.pth>
|-- <trained>
    |-- <NYUDepthv2>
        |-- ...
    |-- <SUNRGBD>
        |-- ...
<datasets>
|-- <DatasetName1>
    |-- <RGB>
    |-- <Depth>
    |-- train.txt
    |-- test.txt
|-- <DatasetName2>
|-- ...
</code></pre>
</details>

---

## Training

You can modify `local_config` in the `train.sh` script to select the model for training:

```bash
bash train.sh
```

The trained weights will be saved under the `checkpoints/XXX` path.

---

## Testing

Similarly, modify `local_config` and the weight path in `eval.sh`:

```bash
bash eval.sh
```

---

## Visualization Inference

```bash
bash infer.sh
```

---

## Calculate FLOPs & Parameters

```bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python benchmark.py --config local_configs.NYUDepthv2.DFormer_Large
```

---

## Test Latency

```bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python utils/latency.py --config local_configs.NYUDepthv2.DFormer_Large
```

*Note: Latency is device-dependent, it is recommended to compare on the same device.*

---

## Acknowledgement

We would like to thank the authors of [DformerV2](https://github.com/VCIP-RGBD/DFormer)] which has significantly accelerated the development of our book.






