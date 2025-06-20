# <p align=center>`DFormer: 用于RGBD语义分割的Transformer`</p>

本仓库为 DFormer 及 DFormerv2 在 RGBD 语义分割任务上的 PyTorch 实现，支持 NYUDepthv2、SUNRGBD 等主流数据集。

---

## 安装环境

```bash
conda create -n dformer python=3.10 -y
conda activate dformer

# CUDA 11.8
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

pip install tqdm opencv-python scipy tensorboardX tabulate easydict ftfy regex
```

---

## 数据集与预训练权重

- **数据集**：  
  默认放在 `datasets` 文件夹下，或用软链接 `ln -s path_to_data datasets`。

  | 数据集 | [GoogleDrive](https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EqActCWQb_pJoHpxvPh4xRgBMApqGAvUjid-XK3wcl08Ug?e=VcIVob) | [百度网盘](https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q) |
  |:---:|:---:|:---:|:---:|

  数据集已将深度 `.npy` 转为 `.png`，并整理了路径及划分文件。

- **预训练权重**：  
  ImageNet-1K 预训练、NYUDepth/SUNRGBD 训练的 DFormer/DFormerv2 权重可在下方下载：

  | 权重 | DFormer | DFormerv2 |
  |-------|-------| -  |
  | 预训练 | [GoogleDrive](https://drive.google.com/drive/folders/1YuW7qUtnguUFkhC-sfqGySrerjK0rZJX?usp=sharing), [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EhTTF_ZofnFIkz2WSDFAiiIBEIubZUpIwDQYwm9Hvxwu8Q?e=x8XumL), [百度网盘](https://pan.baidu.com/s/1JlexzFqMcZOXPNiNkE1zRA?pwd=gct6) | [百度网盘](https://pan.baidu.com/s/1alSvGtGpoW5TRyLxOt1Txw?pwd=i3pn), [HuggingFace](https://huggingface.co/bbynku/DFormerv2/tree/main/DFormerv2/pretrained) |
  | NYUDepthv2 | [GoogleDrive](https://drive.google.com/drive/folders/1P5HwnAvifEI6xiTAx6id24FUCt_i7GH8?usp=sharing), [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/ErAmlYuhS6FCqGQZNGZy0_EBYgJsK3pFTsi2q9g14MEE_A?e=VoKUAf), [百度网盘](https://pan.baidu.com/s/1AkvlsAvJPv21bz2sXlrADQ?pwd=6vuu) | [百度网盘](https://pan.baidu.com/s/1hi_XPCv1JDRBjwk8XN7e-A?pwd=3vym), [HuggingFace](https://huggingface.co/bbynku/DFormerv2/tree/main/DFormerv2/NYU) |
  | SUNRGBD | [GoogleDrive](https://drive.google.com/drive/folders/1b005OUO8QXzh0sJM4iykns_UdlbMNZb8?usp=sharing), [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EiNdyUV486BFvb7H2yJWSCMBElOj-m6EppIy4dSXNX-yNw?e=fu2Che), [百度网盘](https://pan.baidu.com/s/1D6UMiBv6fApV5lafo9J04w?pwd=7ewv) | [百度网盘](https://pan.baidu.com/s/1NUOgzYmrXmwU7XA8RTRYPg?pwd=ytr7), [HuggingFace](https://huggingface.co/bbynku/DFormerv2/tree/main/DFormerv2/SUNRGBD) |

<details>
<summary>建议的 checkpoints 和数据集目录结构如下：</summary>
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

## 训练

可在 `train.sh` 脚本中修改 `local_config` 选择模型进行训练：

```bash
bash train.sh
```

训练权重将保存在 `checkpoints/XXX` 路径下。

---

## 测试

同理，修改 `eval.sh` 中的 `local_config` 和权重路径：

```bash
bash eval.sh
```

---

## 可视化推理

```bash
bash infer.sh
```

---

## 计算 FLOPs & 参数量

```bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python benchmark.py --config local_configs.NYUDepthv2.DFormer_Large
```

---

## 测试延迟

```bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python utils/latency.py --config local_configs.NYUDepthv2.DFormer_Large
```

*注：延迟与设备相关，建议同设备对比。*

---

## 致谢

We would like to thank the authors of [DformerV2](https://github.com/VCIP-RGBD/DFormer)] which has significantly accelerated the development of our book.






