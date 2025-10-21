# Adjustable Spatio-Spectral Hyperspectral Image Compression Network
This repository contains code of the paper [`Adjustable Spatio-Spectral Hyperspectral Image Compression Network`](https://arxiv.org/abs/2507.23447) accepted at IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (Special Section: Al for Remote Sensing). This work has been done at the [Remote Sensing Image Analysis group](https://rsim.berlin/) by [Martin Hermann Paul Fuchs](https://rsim.berlin/team/members/martin-hermann-paul-fuchs), [Behnood Rasti](https://rsim.berlin/team/members/behnood-rasti) and [Begüm Demir](https://rsim.berlin/team/members/begum-demir).

If you use this code, please cite our paper given below:

> M. H. P. Fuchs, B. Rasti and B. Demіr, "[Adjustable Spatio-Spectral Hyperspectral Image Compression Network](https://arxiv.org/abs/2507.23447)", arXiv preprint arXiv:2507.23447 (2025).

```
@article{fuchs2025adjustable,
  title={Adjustable Spatio-Spectral Hyperspectral Image Compression Network},
  author={Fuchs, Martin Hermann Paul and Rasti, Behnood and Demir, Beg{\"u}m},
  journal={arXiv preprint arXiv:2507.23447},
  year={2025}
}
```
This repository contains code that has been adapted from the CompressAI framework https://github.com/InterDigitalInc/CompressAI/.

## Description
With the rapid growth of hyperspectral data archives in remote sensing (RS), the need for efficient storage has become essential, driving significant attention toward learning-based hyperspectral image (HSI) compression. However, a comprehensive investigation of the individual and joint effects of spectral and spatial compression on learning-based HSI compression has not been thoroughly examined yet. Conducting such an analysis is crucial for understanding how the exploitation of spectral, spatial, and joint spatio-spectral redundancies affects HSI compression. To address this issue, we propose Adjustable Spatio-Spectral Hyperspectral Image Compression Network (HyCASS), a learning-based model designed for adjustable HSI compression in both spectral and spatial dimensions. HyCASS consists of six main modules: 1) spectral encoder module; 2) spatial encoder module; 3) compression ratio (CR) adapter encoder module; 4) CR adapter decoder module; 5) spatial decoder module; and 6) spectral decoder module. The modules employ convolutional layers and transformer blocks to capture both short-range and long-range redundancies. Experimental results on three HSI benchmark datasets demonstrate the effectiveness of our proposed adjustable model compared to existing learning-based compression models, surpassing the state of the art by
up to 2.36 dB in terms of PSNR. Based on our results, we establish a guideline for effectively balancing spectral and spatial compression across different CRs, taking into account the spatial resolution of the HSIs. Our code and pre-trained model weights are publicly available at https://git.tu-berlin.de/rsim/hycass.

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

#### Berlin-Urban-Gradient
Go to https://box.hu-berlin.de/f/e4fa78c198bc4d868d30/?dl=1 to download the Berlin-Urban-Gradient dataset, then unzip the archive. Split the tile into patches using [`berlinurbangradient-patchify.ipynb`](datasets/berlinurbangradient-patchify.ipynb) and create the split files using [`datasets/mlretset-split-creation.ipynb`](datasets/berlinurbangradient-split-creation.ipynb).

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
Pre-trained weights are publicly available.

### HySpecNet-11k
https://tubcloud.tu-berlin.de/s/WYJMqZ9TZqySSrt

### Berlin-Urban-Gradient
https://tubcloud.tu-berlin.de/s/2LDG47Zzcy529eG

### MLRetSet
https://tubcloud.tu-berlin.de/s/D4ePyZS6nZ3cozd

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
