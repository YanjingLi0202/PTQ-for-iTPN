# PTQ-for-iTPN

This repo contains the implementation of 8-bit Post-Training Quantization for iTPN (**[Integrally Pre-Trained Transformer Pyramid Networks](https://arxiv.org/pdf/2211.12735.pdf)**). 


## Table of Contents
- [Getting Started](#getting-started)
  - [Install](#install)
  - [Data preparation](#data-preparation)
  - [Run](#run)
- [Results on ImageNet](#results-on-imagenet)
- [Citation](#citation)


## Getting Started

### Install

- Clone this repo.

```bash
git clone https://github.com/YanjingLi0202/PTQ-for-iTPN.git
cd PTQ-for-iTPN
```

- Create a conda virtual environment and activate it.

```bash
conda create -n ptq-for-itpn python=3.7 -y
conda activate ptq-for-itpn
```

- Install PyTorch and torchvision. e.g.,

```bash
conda install pytorch=1.7.1 torchvision cudatoolkit=10.1 -c pytorch
```

### Data preparation

You should download the standard ImageNet Dataset.

```
├── imagenet
│   ├── train
|
│   ├── val
```


### Run

Example: Evaluate quantized iTPN-Base with MinMax quantizer

```bash
python test_quant.py itpn_base </your/path/to/ImageNet> --quant --calib-iter 10 --calib-batchsize 100 --quant-method minmax 
```

- `itpn_base`: model architecture, which can be replaced by `itpn_base`, `itpn_large`. 
- `--quant`: whether to quantize the model.

- `--quant-method`: quantization methods of activations, which can be chosen from `minmax`, `ema`, `percentile` and `omse`.

## Results on ImageNet

This paper employs several current post-training quantization strategies together with our methods, including MinMax, EMA , Percentile and OMSE.

- MinMax uses the minimum and maximum values of the total data as the clipping values; 

- [EMA](https://arxiv.org/abs/1712.05877) is based on MinMax and uses an average moving mechanism to smooth the minimum and maximum values of different mini-batch;

- [Percentile](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Fully_Quantized_Network_for_Object_Detection_CVPR_2019_paper.pdf) assumes that the distribution of values conforms to a normal distribution and uses the percentile to clip. In this paper, we use the 1e-5 percentile because the 1e-4 commonly used in CNNs has poor performance in Vision Transformers. 

- [OMSE](https://arxiv.org/abs/1902.06822) determines the clipping values by minimizing the quantization error. 


The following results are evaluated on ImageNet.

|         Method         | W/A/Attn Bits |   ViT-B   |   ViT-L   |  DeiT-T   |  DeiT-S   |  DeiT-B   |  Swin-T   |  Swin-S   |  Swin-B   |
| :--------------------: | :-----------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|     Full Precision     | 32/32/32  |   84.53   |   85.81   |   72.21   |   79.85   |   81.85   |   81.35   |   83.20   | 83.60 |
|         MinMax         |   8/8/8   |   23.64   |   3.37    |   70.94   |   75.05   |   78.02   |   64.38   |   74.37   | 25.58 |
|     MinMax w/ PTS      |   8/8/8   |   83.31   |   85.03   |   71.61   |   79.17   |   81.20   |   80.51   |   82.71   | 82.97 |
|   MinMax w/ PTS, LIS   | 8/8/**4** | **82.68** |   84.89   |   71.07   |   78.40   |   80.85   |   80.04   |   82.47   | 82.38 |
|          EMA           |   8/8/8   |   30.30   |   3.53    |   71.17   |   75.71   |   78.82   |   70.81   |   75.05   | 28.00 |
|       EMA w/ PTS       |   8/8/8   |   83.49   |   85.10   |   71.66   |   79.09   |   81.43   |   80.52   |   82.81   | 83.01 |
|    EMA w/ PTS, LIS     | 8/8/**4** |   82.57   |   85.08   |   70.91   | **78.53** | **80.90** |   80.02   |   82.56   | 82.43 |
|       Percentile       |   8/8/8   |   46.69   |   5.85    |   71.47   |   76.57   |   78.37   |   78.78   |   78.12   | 40.93 |
|   Percentile w/ PTS    |   8/8/8   |   80.86   |   85.24   |   71.74   |   78.99   |   80.30   |   80.80   |   82.85   | 83.10 |
| Percentile w/ PTS, LIS | 8/8/**4** |   80.22   | **85.17** | **71.23** |   78.30   |   80.02   | **80.46** | **82.67** | 82.79 |
|          OMSE          |   8/8/8   |   73.39   |   11.32   |   71.30   |   75.03   |   79.57   |   79.30   |   78.96   | 48.55 |
|      OMSE w/ PTS       |   8/8/8   |   82.73   |   85.27   |   71.64   |   78.96   |   81.25   |   80.64   |   82.87   | 83.07 |
|    OMSE w/ PTS, LIS    | 8/8/**4** |   82.37   |   85.16   |   70.87   |   78.42   | **80.90** |   80.41   |   82.57   | 82.45 |


