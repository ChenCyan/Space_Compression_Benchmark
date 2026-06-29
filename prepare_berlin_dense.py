"""
从 hymap_berlin.tif 用 stride=40 切 80x80 patch，
归一化到 float32 [0,1]，建立 train/val/test split（80/10/10）。
输出目录：/data/cyl/space_compression/hycass/datasets/berlin-dense/
"""
import csv
import os
import random
import numpy as np
import rasterio

SRC = "/data/cyl/space_compression/hycass/datasets/berlin-urban-gradient/raster/hymap_berlin.tif"
OUT = "/data/cyl/space_compression/hycass/datasets/berlin-dense"
PATCH = 80
STRIDE = 40
SEED = 42

os.makedirs(os.path.join(OUT, "patches"), exist_ok=True)
os.makedirs(os.path.join(OUT, "splits"), exist_ok=True)

print("Loading raster...")
with rasterio.open(SRC) as src:
    cube = src.read().astype(np.float32)  # (111, 3200, 346)

C, H, W = cube.shape
print(f"Raster: {C} bands x {H} x {W}")

# 归一化：与原始 Berlin 数据集一致，使用 DN / 10000
print("Normalizing (dn / 10000)...")
cube = cube / 10000.0

# 切 patch
patch_names = []
idx = 0
print("Cutting patches...")
for r in range(0, H - PATCH + 1, STRIDE):
    for c in range(0, W - PATCH + 1, STRIDE):
        patch = cube[:, r:r + PATCH, c:c + PATCH].astype(np.float32)
        name = f"{idx:04d}.npy"
        np.save(os.path.join(OUT, "patches", name), patch)
        patch_names.append(name)
        idx += 1

print(f"Total patches: {len(patch_names)}")

# 随机 split：80 / 10 / 10
random.seed(SEED)
random.shuffle(patch_names)
n = len(patch_names)
n_train = int(n * 0.8)
n_val   = int(n * 0.1)
train = patch_names[:n_train]
val   = patch_names[n_train:n_train + n_val]
test  = patch_names[n_train + n_val:]

for split_name, items in [("train", train), ("val", val), ("test", test)]:
    with open(os.path.join(OUT, "splits", f"{split_name}.csv"), "w", newline="") as f:
        csv.writer(f).writerows([[x] for x in items])
    print(f"  {split_name}: {len(items)} patches")

print(f"\nDone. Dataset saved to {OUT}")
