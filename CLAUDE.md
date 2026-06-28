# HyCASS Compression Benchmark

## Project
- **Location**: `/data/cyl/space_compression/hycass`
- **Remote**: `ssh huaweiyun` → server 114.116.245.145, user `longjunyu`
- **GPU**: 2× Tesla V100-SXM2-16GB, driver 535.183.01
- **Docker image**: `hycass-torch:2.7` (torch 2.5.1+cu121)
- **Rule**: Only operate on `hycass-torch:*` Docker images. Never touch other users' containers/images.

## Quick start
```bash
ssh huaweiyun 'docker run --rm --gpus all --user root \
  -v /data/cyl/space_compression:/data/cyl/space_compression \
  -w /data/cyl/space_compression/hycass \
  hycass-torch:2.7 bash -c "
    pip install -q pyjpegls lz4 numba -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -m benchmark.runner \
      --dataset datasets/berlin-urban-gradient/ \
      --dataset-type berlin --max-samples 0 --warmup 2 \
      --codecs jpeg2000_lossless ccsds123_lossless \
      --output results/test.csv"'
```

## Directory structure
```
hycass/
├── benchmark/                # Core — unified benchmark framework
│   ├── codecs/               # All codec implementations
│   │   ├── base.py           # Codec ABC + EncodeResult
│   │   ├── jp2_util.py       # glymur JPEG2000 (multi-component)
│   │   ├── jpeg2000.py       # JPEG2000 codec
│   │   ├── jpegls_codec.py   # JPEG-LS (pyjpegls/CharLS)
│   │   ├── klt_dwt.py        # KLT+DWT (sklearn PCA + glymur)
│   │   ├── hycass_codec.py   # All learned models
│   │   ├── ccsds123_py_codec.py  # CCSDS-123 wrapper
│   │   ├── generic_codecs.py # LZ4 + zlib
│   │   └── ccsds123_py/      # CCSDS-123 Python implementation
│   │       ├── compression.py       # Pure Python
│   │       └── compression_fast.py  # Numba-accelerated
│   ├── domain.py             # Unified DN domain
│   ├── runner.py             # CLI entry
│   ├── metrics_unified.py    # MSE/PSNR/PAE/SA
│   └── plots.py
├── datasets/
├── models/
├── pretrained/               # 38 pretrained weights (Berlin only)
├── results/                  # CSV + PDFs
└── Dockerfile / docker-build.sh
```

## 13 Methods (8 classic + 5 learned)
### Classic (no model file, algorithm-based)
| Method | CR control | Library |
|---|---|---|
| CCSDS-123 | None (entropy) | Python + numba |
| JPEG-LS | **NEAR** (0→16) | CharLS C++ |
| JPEG2000 | **cratio** (1.5–20) | OpenJPEG C |
| KLT+DWT | **nc** (28/56/111) × **cratio** | sklearn + glymur |
| LZ4 | None | lz4 C |
| zlib | level 1–9 | Python stdlib |

### Learned (one model per CR)
| Method | CR range |
|---|---|
| CAE1D | 2.0–13.9 |
| CAE3D | 2.0–25.4 |
| SSCNet | 2.0–587 |
| HYCOT | 2.0–55.5 |
| **HyCASS** | 2.1–877 |

## Datasets
- **Berlin-Urban-Gradient**: 111 bands, 80×80, 160 patches (112/32/16), 38 pretrained weights
- **Indian Pines**: 200 bands, 80×80, 9 patches, classic methods only

## Key parameters
- PSNR peak: data actual max (Berlin=10000, Indian=9604)
- DN domain: uint16, `Domain.from_data_dir()` auto-detects from metadata.json
- Patchify: `python patchify_image.py <.mat> <outdir> --patch-size 80`

## Plots
- `_make_split_plots.py` → 7 split figures (lossless/near/lossy)
- `_make_comparison_plots.py` → Berlin vs Indian Pines
- `_make_journal_plots.py` → 300 DPI journal quality

## Critical notes
1. **glymur must be 0.13.8** (0.14.x incompatible with OpenJPEG 2.3.1)
2. JPEG2000 uses multi-component (MCT enabled, negligible effect on 111-band data)
3. **CCSDS-123 near-lossless NOT implemented** (M parameter unused in BrianShTsoi code)
4. Learned methods need pretrained weights — only Berlin has them
5. Docker needs `--user root` for pyjpegls/lz4/numba install

## Slides
- `~/hsi-benchmark-slides/` — Slidev project
- Images: `public/*.png` (PDF→PNG via pdftoppm)
- Theme: default (no sidebar)
