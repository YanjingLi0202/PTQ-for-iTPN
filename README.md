# PTQ-for-iTPN

This repo contains the implementation of 8-bit Post-Training Quantization for iTPN (**[Integrally Pre-Trained Transformer Pyramid Networks](https://arxiv.org/pdf/2211.12735.pdf)**). 


## Table of Contents
- [Getting Started](#getting-started)
  - [Install](#install)
  - [Data preparation](#data-preparation)
  - [Run](#run)
- [Results on ImageNet](#results-on-imagenet)


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

|        Method        | W/A/Attn Bits | iTPN-B-pixel | iTPN-L-pixel |
|:--------------------:|:-------------:|:------------:|:------------:|
| Real-Valued (37epoch)|   32/32/32    |    83.53     |    --.--     |
|        MinMax        |     8/8/8     |    23.64     |    3.37      |
|         EMA          |     8/8/8     |    30.30     |    3.53      |
|      Percentile      |     8/8/8     |    46.69     |    5.85      |
|         OMSE         |     8/8/8     |    73.39     |    11.32     |


