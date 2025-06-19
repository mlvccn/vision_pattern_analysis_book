# CRIS: CLIP-Driven Referring Image Segmentation (CVPR2022)

### 环境依赖

- Python 3.7 + PyTorch 1.10.0（CUDA 10.2）  

- 创建 Conda 环境：  

  ```bash
  conda create -n cris python=3.7
  conda activate cris
  conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
  pip install -r requirements.txt
  ```

### 数据准备

下载以下数据，提取并放置于data文件夹

- https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip
- https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
- https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
- https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip

下载COCO 2014数据集：http://mscoco.org/dataset/

COCO 2014 训练图像（83K/13GB）放入 ./refer/data/images/mscoco/images

### 训练代码

将习题中的自己写的代码 替换 `./model/segmenter.py`中的`CRIS`类。

在训练之前, 请登录你的 wandb 账号，可以用 `wandb login` 或 `wandb login --anonymously` 命令

```
python -u train.py --config config/refcoco/cris_r50.yaml
```

### 测试代码

```
CUDA_VISIBLE_DEVICES=0 python -u test.py \
      --config config/refcoco/cris_r50.yaml \
      --opts TEST.test_split val-test \
             TEST.test_lmdb datasets/lmdb/refcocog_g/val.lmdb
```

### 引用

```
@inproceedings{wang2021cris,
  title={CRIS: CLIP-Driven Referring Image Segmentation},
  author={Wang, Zhaoqing and Lu, Yu and Li, Qiang and Tao, Xunqiang and Guo, Yandong and Gong, Mingming and Liu, Tongliang},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2022}
}
```
