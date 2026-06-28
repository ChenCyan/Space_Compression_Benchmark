#!/usr/bin/env python3
"""Generic image → numpy patches + train/val/test splits.

Usage:
    python patchify_image.py <mat_file> <output_dir> [--patch-size 64] [--stride 32]

Supports:
    - .mat files (scipy.io.loadmat, auto-detects first 3-D numeric array)
    - .npy files
    - .tif/.tiff files (via rasterio)

Output:
    output_dir/
        patches/    000.npy ...  (float32 [0,1])
        splits/     train.csv, val.csv, test.csv
        metadata.json
"""

import argparse
import csv
import json
import os
import random
import sys

import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load a hyperspectral image from various formats. Returns (C, H, W) float32."""

    ext = os.path.splitext(path)[1].lower()

    if ext in (".mat",):
        import scipy.io as sio
        mat = sio.loadmat(path)
        # Find the largest 3-D numeric array
        best = None
        for key, val in mat.items():
            if key.startswith("__"):
                continue
            if isinstance(val, np.ndarray) and val.ndim == 3:
                if best is None or val.size > best.size:
                    best = val
        if best is None:
            raise ValueError(f"No 3-D array found in {path}. Keys: {list(mat.keys())}")
        arr = best.astype(np.float64)

    elif ext in (".npy",):
        arr = np.load(path).astype(np.float64)

    elif ext in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float64)  # rasterio returns (C, H, W)

    else:
        raise ValueError(f"Unsupported format: {ext}")

    # Detect axis order: if last dim is small (likely bands), transpose to (C, H, W)
    if arr.ndim == 3 and arr.shape[0] > arr.shape[2]:
        # Could be (H, W, C) → (C, H, W)
        pass  # keep as-is if already (C, H, W)

    # If shape is (H, W, C), transpose
    if arr.ndim == 3 and min(arr.shape) == arr.shape[2] and arr.shape[2] < arr.shape[0]:
        arr = arr.transpose(2, 0, 1)  # (H,W,C) → (C,H,W)

    return arr


def patchify(
    arr: np.ndarray,
    patch_size: int,
    stride: int,
) -> list[np.ndarray]:
    """Sliding-window patchify a (C, H, W) array. Returns list of (C, patch, patch)."""
    C, H, W = arr.shape
    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = arr[:, y:y + patch_size, x:x + patch_size]
            patches.append(patch)
    return patches


def main():
    ap = argparse.ArgumentParser(description="Patchify a hyperspectral image")
    ap.add_argument("input", help="Path to .mat / .npy / .tif file")
    ap.add_argument("output_dir", help="Output directory for patches + splits")
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=None,
                    help="Stride (default: patch_size, i.e. no overlap)")
    ap.add_argument("--seed", type=int, default=10587)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    args = ap.parse_args()
    stride = args.stride if args.stride is not None else args.patch_size

    print(f"Loading {args.input} ...")
    arr = load_image(args.input)
    C, H, W = arr.shape
    print(f"  Shape: ({C}, {H}, {W}), dtype={arr.dtype}")

    # Global min-max normalization to [0, 1]
    vmin, vmax = float(arr.min()), float(arr.max())
    print(f"  Range: [{vmin:.0f}, {vmax:.0f}]")
    arr_norm = (arr - vmin) / max(vmax - vmin, 1e-12)
    arr_norm = arr_norm.astype(np.float32)

    print(f"  Patchifying ({args.patch_size}×{args.patch_size}, stride={stride}) ...")
    patches = patchify(arr_norm, args.patch_size, stride)
    print(f"  → {len(patches)} patches")

    # Save patches
    patches_dir = os.path.join(args.output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)
    fnames = []
    for i, p in enumerate(patches):
        fname = f"{i:04d}.npy"
        np.save(os.path.join(patches_dir, fname), p)
        fnames.append(fname)
    print(f"  Saved to {patches_dir}/")

    # Train/val/test split
    rng = random.Random(args.seed)
    indices = list(range(len(fnames)))
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    splits_dir = os.path.join(args.output_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    for split_name, idxs in [
        ("train", indices[:n_train]),
        ("val", indices[n_train:n_train + n_val]),
        ("test", indices[n_train + n_val:]),
    ]:
        with open(os.path.join(splits_dir, f"{split_name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            for i in sorted(idxs):
                w.writerow([fnames[i]])
        print(f"  {split_name}: {len(idxs)} patches")

    # Metadata
    meta = {
        "source": args.input,
        "shape": [C, H, W],
        "vmin": vmin, "vmax": vmax,
        "patch_size": args.patch_size,
        "stride": stride,
        "n_patches": len(patches),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. Dataset ready at {args.output_dir}/")


if __name__ == "__main__":
    main()
