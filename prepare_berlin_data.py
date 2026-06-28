#!/usr/bin/env python3
"""
Berlin-Urban-Gradient dataset preparation: patchify + train/val/test splits.

This script replaces the two Jupyter notebooks:
  - datasets/berlinurbangradient-patchify.ipynb
  - datasets/berlinurbangradient-split-creation.ipynb

Run from the hycass project root after downloading the zip:
  1. Download: curl -L -o datasets/berlin_raw.zip "https://box.hu-berlin.de/f/e4fa78c198bc4d868d30/?dl=1"
  2. Unzip:    unzip datasets/berlin_raw.zip -d datasets/berlin-urban-gradient/
  3. Run:      python prepare_berlin_data.py datasets/berlin-urban-gradient/
"""

import csv
import glob
import os
import random
import sys

import numpy as np


def patchify(dataset_path: str):
    """Load the TIFF, min-max normalize, split into 80x80 non-overlapping patches.

    Expects the raster at: {dataset_path}/raster/hymap_berlin.tif
    Saves patches to:      {dataset_path}/patches/000.npy, 001.npy, ...
    """
    tif_path = os.path.join(dataset_path, "raster", "hymap_berlin.tif")
    out_dir = os.path.join(dataset_path, "patches")
    os.makedirs(out_dir, exist_ok=True)

    # Try multiple TIFF loaders (rasterio, tifffile, imageio).
    data = None
    for lib_name, loader in _TIFF_LOADERS:
        try:
            data = loader(tif_path)
            print(f"  Loaded TIFF via {lib_name}, shape={data.shape}, dtype={data.dtype}")
            break
        except Exception:
            continue

    if data is None:
        raise RuntimeError(
            "Cannot read the GeoTIFF. Install one of: rasterio, tifffile, imageio.\n"
            "  pip install rasterio"
        )

    # Crop: remove 13 pixels from each edge (matching original notebook).
    data_cropped = data[:, :, 13:-13]
    print(f"  After crop: shape={data_cropped.shape}")

    # Global min-max normalization to [0, 1].
    vmin = data_cropped.min()
    vmax = data_cropped.max()
    data_norm = (np.float32(data_cropped) - vmin) / (vmax - vmin)
    print(f"  Normalized: min={data_norm.min():.4f}, max={data_norm.max():.4f}")

    # Split into non-overlapping 80×80 patches.
    C, H, W = data_norm.shape
    patch_h, patch_w = 80, 80
    patches = data_norm.reshape(
        C,
        H // patch_h, patch_h,
        W // patch_w, patch_w,
    ).transpose(1, 3, 0, 2, 4).reshape(-1, C, patch_h, patch_w)

    print(f"  Patches: {len(patches)} × ({C}, {patch_h}, {patch_w})")

    for i, patch in enumerate(patches):
        np.save(os.path.join(out_dir, f"{i:03d}.npy"), patch)

    print(f"  Saved {len(patches)} patches to {out_dir}/")
    return len(patches)


def create_splits(dataset_path: str, seed: int = 10587):
    """Create train/val/test CSV splits (70/20/10) matching repo's notebook."""
    patches_dir = os.path.join(dataset_path, "patches")
    splits_dir = os.path.join(dataset_path, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    # Gather patch filenames
    files = sorted(glob.glob(os.path.join(patches_dir, "*.npy")))
    files = [os.path.basename(f) for f in files]
    print(f"  Found {len(files)} .npy patches")

    # Shuffle with fixed seed
    rng = random.Random(seed)
    rng.shuffle(files)

    # Split 70/20/10
    n = len(files)
    n_train = int(n * 0.7)
    n_val = int(n * 0.2)
    train = sorted(files[:n_train])
    val = sorted(files[n_train : n_train + n_val])
    test = sorted(files[n_train + n_val :])

    for name, lst in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(splits_dir, f"{name}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            for item in lst:
                w.writerow([item])
        print(f"  {name}: {len(lst)} patches → {path}")

    return train, val, test


# --------------------------------------------------------------------------
# TIFF loaders (tried in order of preference)
# --------------------------------------------------------------------------

def _load_rasterio(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.read()  # (bands, H, W)


def _load_tifffile(path):
    """tifffile returns (H, W, C) for multi-page → transpose to (C, H, W)."""
    import tifffile
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    return arr.transpose(2, 0, 1)


def _load_imageio(path):
    import imageio.v3 as iio
    arr = iio.imread(path)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    return arr.transpose(2, 0, 1)


_TIFF_LOADERS = [
    ("rasterio", _load_rasterio),
    ("tifffile", _load_tifffile),
    ("imageio", _load_imageio),
]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prepare_berlin_data.py <dataset_path>")
        print("Example: python prepare_berlin_data.py datasets/berlin-urban-gradient/")
        sys.exit(1)

    ds_path = sys.argv[1].rstrip("/")

    print("=" * 50)
    print("Step 1: Patchify")
    print("=" * 50)
    n_patches = patchify(ds_path)

    print()
    print("=" * 50)
    print("Step 2: Train/Val/Test Splits")
    print("=" * 50)
    create_splits(ds_path)

    print()
    print("Done! Dataset ready for benchmarking.")
    print(f"  python -m benchmark.runner --dataset {ds_path}/ --dataset-type berlin --codecs ...")
