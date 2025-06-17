# ACARNet复现

论文：Actor-Context-Actor Relation Network for Spatio-Temporal Action Localization(CVPR 2021)

## 数据集准备

You could refer to [PySlowFast DATASET.md](https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/DATASET.md) for AVA dataset preparation. 

To simplify the process, we provide the [clipped video data](https://pan.baidu.com/s/1v2LrWLsncycMjuPa6dTOIA?pwd=wqhy), so you can directly start processing from the frame extraction step.

### Extract frames

```bash
IN_DATA_DIR="path/to/ava/videos_15min"  #数据集视频路径
OUT_DATA_DIR="path/to/ava/frames" #输出数据集视频帧路径

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  video_name=${video##*/}

  if [[ $video_name = *".webm" ]]; then
    video_name=${video_name::-5}
  else
    video_name=${video_name::-4}
  fi

  out_video_dir=${OUT_DATA_DIR}/${video_name}/
  mkdir -p "${out_video_dir}"

  out_name="${out_video_dir}/${video_name}_%06d.jpg"

  ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
done
```

AVA dataset with the following structure:

```
ava
|_ frames
|  |_ [video name 0]
|  |  |_ [video name 0]_000001.jpg
|  |  |_ [video name 0]_000002.jpg
|  |  |_ ...
|  |_ [video name 1]
|     |_ [video name 1]_000001.jpg
|     |_ [video name 1]_000002.jpg
|     |_ ...
```

### Download annotations

Download annotations from [here](https://pan.baidu.com/s/12RmMFX3-a246QTtUccogpQ?pwd=f895). And move all the files in the annotations folder to the “ACAR-Net/annotations” folder.

### Download pretrained models

Download pretrained models from [here](https://pan.baidu.com/s/10jUEqps57OtUx2cpnE4vvw?pwd=xyk3 ). And move all the files in the pretrained folder to the “ACAR-Net/pretrained” folder.

## 环境安装

```bash
conda create -n acar python=3.8
conda activate acar
cd ACAR-Net
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
pip install numpy==1.19 -i https://pypi.mirrors.ustc.edu.cn/simple
```

## 运行命令

修改configs/AVA/SLOWFAST_R50_ACAR_HR2O.yaml中的配置：result_path、train.root_path和val.root_path

```
python main.py --config configs/AVA/SLOWFAST_R50_ACAR_HR2O.yaml --nproc_per_node 1 --backend nccl --master_addr 127.0.0.1 --master_port 31114
```
--nproc_per_node 为1表示单卡，大于1为多卡。

根据自己实际情况修改train.batch_size和val.batch_size，train.batch_size=1大约需要7GB显存，train.batch_size=2大约需要12GB显存，train.batch_size=3大约需要16GB显存，train.batch_size=4大约需要20GB显存。

训练时整个batchsize=nproc_per_node*train.batch_size
## About Paper

Please cite with the following Bibtex code:

```
@inproceedings{pan2021actor,
  title={Actor-context-actor relation network for spatio-temporal action localization},
  author={Pan, Junting and Chen, Siyu and Shou, Mike Zheng and Liu, Yu and Shao, Jing and Li, Hongsheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={464--474},
  year={2021}
}
```
