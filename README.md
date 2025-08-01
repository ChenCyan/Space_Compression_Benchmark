# Adjustable Spatio-Spectral Hyperspectral Image Compression Network
This repository contains code of the paper [`Adjustable Spatio-Spectral Hyperspectral Image Compression Network`](https://arxiv.org/abs/2507.23447) submitted to IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (Special Section: Al for Remote Sensing). This work has been done at the [Remote Sensing Image Analysis group](https://rsim.berlin/) by [Martin Hermann Paul Fuchs](https://rsim.berlin/team/members/martin-hermann-paul-fuchs), [Behnood Rasti](https://rsim.berlin/team/members/behnood-rasti) and [Begüm Demir](https://rsim.berlin/team/members/begum-demir).

If you use this code, please cite our paper given below:

> M. H. P. Fuchs, B. Rasti and B. Demіr, "[Adjustable Spatio-Spectral Hyperspectral Image Compression Network](https://arxiv.org/abs/2507.23447)", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2025, under review.

```
@article{Fuchs:2025,
    author={M. H. P. {Fuchs}, B. {Rasti} and B. {Demіr}},
    journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
    title={Adjustable Spatio-Spectral Hyperspectral Image Compression Network}, 
    year={2025, under review}
}
```
This repository contains code that has been adapted from the CompressAI [\[1\]](#2-compressai) framework https://github.com/InterDigitalInc/CompressAI/.

## Description
With the rapid growth of hyperspectral data archives in remote sensing (RS), the need for efficient storage has become essential, driving significant attention toward learning-based hyperspectral image (HSI) compression. However, a comprehensive investigation of the individual and joint effects of spectral and spatial compression on learning-based HSI compression has not been thoroughly examined yet. Conducting such an analysis is crucial for understanding how the exploitation of spectral, spatial, and joint spatio-spectral redundancies affects HSI compression. To address this issue, we propose Adjustable Spatio-Spectral Hyperspectral Image Compression Network (HyCASS), a learning-based model designed for adjustable HSI compression in both spectral and spatial dimensions. HyCASS consists of six main modules: 1) spectral encoder; 2) spatial encoder; 3) compression ratio (CR) adapter encoder; 4) CR adapter decoder; 5) spatial decoder; and 6) spectral decoder module. The modules employ convolutional layers and transformer blocks to capture both short-range and long-range redundancies. Experimental results on two HSI benchmark datasets demonstrate the effectiveness of our proposed adjustable model compared to existing learning-based compression models. Based on our results, we establish a guideline for effectively balancing spectral and spatial compression across different CRs, taking into account the spatial resolution of the HSIs. Our code and pre-trained model weights are publicly available at https://git.tu-berlin.de/rsim/hycass.

## Setup
The code in this repository is tested with `Ubuntu 22.04 LTS` and `Python 3.13.2`.

### Dependencies
All dependencies are listed in the [`requirements.txt`](requirements.txt) and can be installed via the following command:
```
pip install -r requirements.txt
```

### Datasets

#### HySpecNet-11k
Follow the instructions on https://hyspecnet.rsim.berlin to download, extract and preprocess the HySpecNet-11k dataset.

#### MLRetSet
Go to https://www.doi.org/10.17605/OSF.IO/H2T8U to download the MLRetSet dataset, then unzip the archives and create the split files using [`datasets/mlretset-split-creation.ipynb`](datasets/mlretset-split-creation.ipynb).

## Usage

### Train
The [`train.py`](train.py) expects the following command line arguments:
| Parameter | Description | Default |
| :- | :- | :- |
| `--devices` | Devices to use, e.g. `cpu` or `0` or `0,2,5,7` | `0` |
| `--train-batch-size` | Training batch size | `2` |
| `--val-batch-size` | Validation batch size | `2` |
| `-n` | Data loaders threads | `4` |
| `-d` | Path to dataset | `./datasets/hyspecnet-11k/` |
| `--mode` | Dataset split difficulty | `easy` |
| `--transform` | Dataset transformation, e.g. `random_16x16` | `None` |
| `-m` | Model architecture | `hycass_cr202_spatial2x_n128` |
| `--loss` | Loss | `mse` |
| `-e` | Number of epochs | `200` |
| `-lr` | Learning rate | `1e-4` |
| `--save-dir` | Directory to save results | `./results/` |
| `--experiment-name` | Name of experiment | `trial` |
| `--seed` | Set random seed for reproducibility | `10587` |
| `--clip-max-norm` | Gradient clipping max norm | `1.0` |
| `--checkpoint` | Path to a checkpoint to resume training | `None` |

Specify the parameters in the [`train.sh`](train.sh) file and then execute the following command:
```console
./train.sh
```
Or run the python code directly through the console:
```console
python train.py \
    --devices 0 \
    --train-batch-size 16 \
    --val-batch-size 16 \
    --num-workers 4 \
    --learning-rate 1e-4 \
    --mode easy \
    --model hycass_cr202_spatial2x_n128 \
    --loss mse \
    --epochs 200
```
### Test
The test is automatically executed after training.

## Pre-Trained Weights
Pre-trained weights are publicly available and should be downloaded into the [`./results/weights/`](results/weights/) folder.

| Method | Model | Compression Ratio | PSNR | Download Link |
| :----- | :---- | :--- | :--- | :------------ |
| 1D-CAE [\[2\]](#3-1d-convolutional-autoencoder-1d-cae) | `cae1d_cr32` | 28.86 | 48.95 dB | [cae1d_1bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/ew2jr67yro7cj3x/download/cae1d_1bpppc.pth.tar) |
| | `cae1d_cr16` | 15.54 | 52.38 dB | [cae1d_2bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/Ae35EBRado8QSmk/download/cae1d_2bpppc.pth.tar) |
| | `cae1d_cr8` | 7.77 | 53.90 dB | [cae1d_4bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/ZNeXycsssRdYZ5m/download/cae1d_4bpppc.pth.tar) |
| | `cae1d_cr4` | 3.96 | 54.85 dB | [cae1d_8bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/GpmXDAWEeo2nG5w/download/cae1d_8bpppc.pth.tar) |
| SSCNet [\[3\]](#4-spectral-signals-compressor-network-sscnet) | `sscnet_cr32` | 32.00 | 43.24 dB | [sscnet_1bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/wPwbMKYJAmXxLRX/download/sscnet_1bpppc.pth.tar) |
| | `sscnet_cr16` | 15.84 | 43.60 dB | [sscnet_2bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/H9Yg8n8rzxGMe2Z/download/sscnet_2bpppc.pth.tar) |
| | `sscnet_cr8` | 8.00 | 43.69 dB | [sscnet_4bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/WQ65aCDxgedQYxZ/download/sscnet_4bpppc.pth.tar) |
| | `sscnet_cr4` | 3.96 | 43.29 dB | [sscnet_8bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/5kiQ8ZLRnkpbSg6/download/sscnet_8bpppc.pth.tar) |
| 3D-CAE [\[4\]](#5-3d-convolutional-auto-encoder-3d-cae) | `cae3d_cr32` | 31.69 | 39.06 dB | [cae3d_1bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/QDfARfWL3Pab3xK/download/cae3d_1bpppc.pth.tar) |
| | `cae3d_cr16` | 15.84 | 39.54 dB | [cae3d_2bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/dD3qtjrgzJxmymP/download/cae3d_2bpppc.pth.tar) |
| | `cae3d_cr8` | 7.92 | 39.69 dB | [cae3d_4bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/CmTdQzcE3x9pEEJ/download/cae3d_4bpppc.pth.tar) |
| | `cae3d_cr4` | 3.96 | 39.94 dB | [cae3d_8bpppc.pth.tar](https://tubcloud.tu-berlin.de/s/DpqKJdMbojF3CLx/download/cae3d_8bpppc.pth.tar) |
| HyCoT | `hycot_cr32` | 28.86 | 50.26 dB | [hycot_cr32.pth.tar](https://tubcloud.tu-berlin.de/s/QsT3An5WTPbDXQS/download/hycot_cr32.pth.tar) |
| | `hycot_cr16` | 15.54 | 53.20 dB | [hycot_cr16.pth.tar](https://tubcloud.tu-berlin.de/s/5jGeG29kTJfHX58/download/hycot_cr16.pth.tar) |
| | `hycot_cr8` | 7.77 | 55.38 dB | [hycot_cr8.pth.tar](https://tubcloud.tu-berlin.de/s/As8yaM3k2isjX92/download/hycot_cr8.pth.tar) |
| | `hycot_cr4` | 3.96 | 56.29 dB | [hycot_cr4.pth.tar](https://tubcloud.tu-berlin.de/s/jeaSXYQHN7ki3mE/download/hycot_cr4.pth.tar) |

## Authors
**Martin Hermann Paul Fuchs**
https://rsim.berlin/team/members/martin-hermann-paul-fuchs

## License
The code in this repository is licensed under the **MIT License**:
```
MIT License

Copyright (c) 2025 Martin Hermann Paul Fuchs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## References
### [1] [CompressAI](https://doi.org/10.48550/arXiv.2011.03029)

### [2] [1D-Convolutional Autoencoder (1D-CAE)](https://doi.org/10.5194/isprs-archives-XLIII-B1-2021-15-2021)

### [3] [Spectral Signals Compressor Network (SSCNet)](https://doi.org/10.3390/rs14102472)

### [4] [3D Convolutional Auto-Encoder (3D-CAE)](https://doi.org/10.1117/1.JEI.30.4.041403)

### [5] [HyCoT: A Transformer-Based Autoencoder for Hyperspectral Image Compression](https://doi.org/10.1109/WHISPERS65427.2024.10876514)