# 基于CLIP的指代图像分割（CVPR2022）

### Environment Requirements

- Python 3.7 + PyTorch 1.10.0 (CUDA 10.2)  

- Create Conda environment:  

  ```bash
  conda create -n cris python=3.7
  conda activate cris
  conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
  pip install -r requirements.txt
  ```

### Data Preparation

Download the following data, extract and place them in the data folder:

- https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip
- https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
- https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
- https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip

Download the COCO 2014 dataset: http://mscoco.org/dataset/

Place COCO 2014 training images (83K/13GB) in ./refer/data/images/mscoco/images

### Training

Replace the `CRIS` class in `./model/segmenter.py` with your own code from the exercises.

Before training, please log in to your wandb account using `wandb login` or `wandb login --anonymously`.

```
python -u train.py --config config/refcoco/cris_r50.yaml
```

### Testing

```
CUDA_VISIBLE_DEVICES=0 python -u test.py \
      --config config/refcoco/cris_r50.yaml \
      --opts TEST.test_split val-test \
             TEST.test_lmdb datasets/lmdb/refcocog_g/val.lmdb
```

### Citation

```
@inproceedings{wang2021cris,
  title={CRIS: CLIP-Driven Referring Image Segmentation},
  author={Wang, Zhaoqing and Lu, Yu and Li, Qiang and Tao, Xunqiang and Guo, Yandong and Gong, Mingming and Liu, Tongliang},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2022}
}
```
