# PRM: A Pixel-Region-Matching Approach for Fast Video Object Segmentation

[![codecov](https://codecov.io/gh/li-plus/PRM/branch/master/graph/badge.svg?token=HOxeP0BM8P)](https://codecov.io/gh/li-plus/PRM) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

A PyTorch implementation of our paper PRM: A Pixel-Region-Matching Approach for Fast Video Object Segmentation.

## Getting Started

This project is developed and tested in the following environments. 

+ Ubuntu 16.04
+ CUDA 9.0.176
+ GeForce GTX 1080 Ti
+ python 3.7

First, clone this project to your local environment. Here we set the environment variable `ROOT` for better clarity, but it is not necessary.

```bash
git clone https://github.com/li-plus/PRM.git
cd PRM/
export ROOT=$(pwd)
```

A conda virtual environment is recommended for the python environment. To create and activate a conda environment with python 3.7, run

```bash
conda create --name prm python=3.7
conda activate prm
```

Then install python dependencies for this project.

```bash
pip install -r requirements.txt
```

## Datasets Preparation

### Downloading

Download [DAVIS 2016](https://davischallenge.org/davis2016/code.html), [DAVIS 2017](https://davischallenge.org/davis2017/code.html) train-val and test-dev, and [YouTube-VOS 2018](https://youtube-vos.org/dataset/vos/) datasets from their official websites. Note that for DAVIS 2016 or DAVIS 2017, only the version of 480p resolution is needed. It is recommended to follow the directory structure below.

```
PRM
└── datasets
    ├── DAVIS2016
    │   ├── Annotations
    │   ├── ImageSets
    │   └── JPEGImages
    ├── DAVIS2017
    │   ├── Annotations
    │   ├── ImageSets
    │   └── JPEGImages
    ├── DAVIS2017_test
    │   ├── Annotations
    │   ├── ImageSets
    │   └── JPEGImages
    └── YouTubeVOS
        ├── train
        │   ├── Annotations
        │   └── JPEGImages
        └── valid
            ├── Annotations
            └── JPEGImages
```

If you save the datasets into another directory, you may need to make a soft link, or manually adjust the path to your datasets in configuration file `$ROOT/src/dataloader/dataset_config.json`.

### Indexing

To simplify the codes for data loading, before training our model, we generate training and validation splits for all datasets.

```bash
cd $ROOT/src/dataloader/
python generate_split.py --dataset davis2016 --dataset-type train --output-dir ../../splits/
python generate_split.py --dataset davis2016 --dataset-type val --output-dir ../../splits/
python generate_split.py --dataset davis2017 --dataset-type train --output-dir ../../splits/
python generate_split.py --dataset davis2017 --dataset-type val --output-dir ../../splits/
python generate_split.py --dataset davis2017 --dataset-type test --output-dir ../../splits/
python generate_split.py --dataset youtube --dataset-type train --output-dir ../../splits/
python generate_split.py --dataset youtube --dataset-type test --output-dir ../../splits/
```

Since we have already generated the splits, now we can specify different splits to train or evaluate on different datasets.

## Evaluation

We provide our pretrained models on DAVIS 2017 and DAVIS 2016.

```bash
cd $ROOT/models/
mkdir -p pretrain/ && cd pretrain/
wget https://cloud.tsinghua.edu.cn/f/72af8aca3a77463c9222/?dl=1 -O prm_davis16.pt
wget https://cloud.tsinghua.edu.cn/f/d2bdeb900dc042ec807b/?dl=1 -O prm_davis17.pt
```

To evaluate a given model on a specific dataset, please specify the path to model and the corresponding split file. For example, to evaluate the pretrained model on DAVIS 2017, run

```bash
cd $ROOT/src/
CUDA_VISIBLE_DEVICES=0 python evaluate.py --split-val ../splits/split_davis2017_val.json \
    --resume ../models/pretrain/prm_davis17.pt --save-dir ../models/pretrain/results/davis17/ 
```

The script will generate separate mask results for each object and save them in the given `--save-dir`. We then merge the separate results into unified masks, and delete the separate results to save our disk space. The final results will be saved in `--save-dir`.

```bash
python merge_masks.py --dir ../models/pretrain/results/davis17/ \
    --save-dir ../models/pretrain/results/davis17.final/
rm -r ../models/pretrain/results/davis17/
```

To evaluate the performance on DAVIS 2017, we use the [official evaluation codes for DAVIS 2017](https://github.com/davisvideochallenge/davis2017-evaluation). Please follow its instructions to evaluate the final results.

Similarly, to evaluate our pretrained model on DAVIS 2016, run

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --split-val ../splits/split_davis2016_val.json \
    --resume ../models/pretrain/prm_davis16.pt --save-dir ../models/pretrain/results/davis16/ \
    --input-height 360 --input-width 640

python merge_masks.py --dir ../models/pretrain/results/davis16/ \
    --save-dir ../models/pretrain/results/davis16.final/

rm -r ../models/pretrain/results/davis16/
```

Please use the [official evaluation codes for DAVIS 2016](https://github.com/davisvideochallenge/davis-2017) to evaluate the final mask results.

## Training

### YouTube-VOS 2018

Firstly, we pretrain our model on YouTube-VOS.

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --model-dir ../models/youtube --batch-size 16 \
    --split-train ../splits/split_youtube_train.json \
    --input-height 360 --input-width 640
```

The training program automatically uses all available GPUs for data parallel. If you have more than 2 GPUs, please specify `CUDA_VISIBLE_DEVICES`.

This process takes roughly 24 hours on our machine. We save 2 checkpoints every epoch, and a total of 40 checkpoints will be saved after this process is done.

### DAVIS 2017

For better performance, we further train our model only on DAVIS 2017 based on the best pretrained model. To get the best checkpoint on DAVIS 2017, we evaluate all checkpoints with the following scripts.

```bash
cd $ROOT/script/
bash test_all.sh \
    --split split_davis2017_val.json --gpu 0,1,2,3 --max-thread 20 \
    --dir ../models/youtube/ --matching cosine --keep-topk 32 \
    --fix-margin 0.5 --input-heigh 360 --input-width 640
```

Now we get the best pretrained checkpoint on DAVIS 2017, say 80999.pt. Then we resume the best checkpoint and train our model for DAVIS 2017.

```bash
cd $ROOT/src/
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --max-epoch 60 --model-dir ../models/davis17 \
    --resume ../models/youtube/checkpoints/80999.pt \
    --split-train ../splits/split_davis2017_train.json \
    --base-lr 1e-6 --save-step 640 --lr-decay-step 1920 \
    --input-height 360 --input-width 640
```

To evaluate all checkpoints, run

```bash
cd $ROOT/script/
bash test_all.sh \
    --split split_davis2017_val.json --gpu 0,1,2,3 --max-thread 20 \
    --dir ../models/youtube/ --matching cosine --keep-topk 32 \
    --fix-margin 0.5 --input-heigh 360 --input-width 640
```

### DAVIS 2016

Similarly, we evaluate all checkpoints on DAVIS 2016.

```bash
cd $ROOT/script/
bash test_all.sh \
    --split split_davis2016_val.json --gpu 0,1,2,3 --max-thread 20 \
    --dir ../models/youtube/ --matching cosine --keep-topk 32 \
    --fix-margin 0.5 --input-heigh 360 --input-width 640
```

Now we get the best pretrained checkpoint on DAVIS 2016, say 80999.pt. Then we train our model based on the best checkpoint.

```bash
cd $ROOT/src/
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --model-dir ../models/davis16 --max-epoch 100 \
    --split-train ../splits/split_davis2016_train.json \
    --resume ../models/youtube/checkpoints/80999.pt \
    --base-lr 1e-6 --save-step 130 --lr-decay-step 650 \
    --input-height 360 --input-width 640
```

To evaluate all checkpoints, use `$ROOT/script/test_all.sh` as well.

## Demo

TODO

## Citation

If you find our codes or paper helpful, please consider citing.

TODO
