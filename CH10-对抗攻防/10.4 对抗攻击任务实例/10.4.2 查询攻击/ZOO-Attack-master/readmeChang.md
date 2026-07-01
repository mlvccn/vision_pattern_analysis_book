# ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks (AIsec 2017)

ZOO（Zeroth Order Optimization）是一种基于零阶优化的深度神经网络（DNNs）攻击方法。我们提出了一种仅需获取目标DNN输入（图像）和输出（置信度分数）即可实施的高效黑盒攻击方案。该攻击被构建为一个优化问题（与Carlini-Wagner攻击类似），并针对黑盒场景设计了新型损失函数。我们采用零阶随机坐标下降法直接对目标DNN进行优化，结合降维处理、分层攻击和重要性采样技术以提升攻击效率。该方法无需依赖迁移性或替代模型。

### Installation

The code was tested on a Conda environment installed on Ubuntu 18.04.

Install [Conda](https://docs.conda.io/en/latest/miniconda.html) and then create an environment as follows:

## Environments
* python == 3.7.11
* pytorch == 1.8.0
* torchvision == 0.8.0
* numpy == 1.21.2
* pandas == 1.3.5
* opencv-python == 4.5.4.60
* scipy == 1.7.3
* pillow == 8.4.0
* pretrainedmodels == 0.7.4
* tqdm == 4.62.3
* imageio == 2.6.1

## Preparation

Prepare the MNIST and CIFAR-10 data and models for attack:

```
python3 train_models.py
```

To download the inception model:

```
python3 setup_inception.py
```

To prepare the ImageNet dataset, download and unzip the following archive:

[ImageNet Test Set](http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/img.tar.gz)


and put the `imgs` folder in `../imagenetdata`. This path can be changed
in `setup_inception.py`.

## Attack

An unified attack interface, `test_all.py` is provided. Run `python3 test_all.py -h`
to get a list of arguments and help.

The following are some examples of attacks:

Run ZOO black-box targeted attack, on the mnist dataset with 200 images, with
ZOO-ADAM solver, search for best regularization constant for 9 iterations, and
save attack images to folder `black_results`. To run on the CIFAR-10 dataset,
replace 'mnist' with 'cifar10'.

```
python3 test_all.py -a black -d mnist -n 200 --solver adam -b 9 -s "black_results"
```

Run Carlini and Wagner's white-box targeted attack, on the mnist dataset with
200 images, using the Z (logits) value in objective (only available in
white-box setting), search for best regularization constant for 9 iterations,
and save attack images to folder `white_results`.

```
python3 test_all.py -a white -d mnist -n 200 --use_zvalue -b 9 -s "white_results"
```

Run ZOO black-box *untargeted* attack, on the imagenet dataset with 150 images, with ZOO-ADAM
solver, do not binary search the regularization parameter (i.e., search only 1
time), and set the initial regularization parameter to a fixed value (10.0). Use
attack-space dimension reduction with image resizing, and reset ADAM states
when the first attack is found.  Run a maximum of 1500 iterations, and print
out loss every 10 iterations. Save attack images to folder `imagenet_untargeted`.

```
python3 test_all.py --untargeted -a black -d imagenet -n 150 --solver adam -b 1 -c 10.0 --use_resize --reset_adam -m 1500 -p 10 -s "imagenet_untargeted"
```

Run ZOO black-box targeted attack, on the imagenet dataset, with the 69th image
only.  Set the regularization parameter to 10.0 and do not binary search. Use
attack-space dimension reduction and hierarchical attack with image resizing,
and reset ADAM states when the first attack is found.  Run a maximum of 20000
iterations, and print out loss every 10 iterations. Save attack images to
folder `imagenet_all_tricks_img69`.


```
python3 test_all.py -a black --solver adam -d imagenet -f 69 -n 1 -c 10.0 --use_resize --reset_adam -m 20000 -p 10 -s "imagenet_all_tricks_img69"
```

Importance sampling is on by default for ImageNet data, and can be turned off by
`--uniform` option. To change the hierarchical attack dimension scheduling,
change `l2_attack_black.py`, near line 580.

## Acknowledgement

We would like to thank the authors of [Zoo]( https://github.com/IBM/ZOO-Attack) ,providing code for academic practice

