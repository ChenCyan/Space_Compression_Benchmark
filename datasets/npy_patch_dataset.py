"""Generic dataset for any directory of .npy patches with CSV splits.

Reuses the exact same interface as BerlinUrbanGradient / HySpecNet11k,
so the benchmark runner needs zero changes for new datasets.

Directory structure:
    root_dir/
        patches/
            000.npy
            001.npy
            ...
        splits/
            train.csv      (one filename per row)
            val.csv
            test.csv

Each .npy file is a float32 array of shape (C, H, W) in [0, 1].
"""

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyPatchDataset(Dataset):
    """Generic (C, H, W) numpy-patch dataset."""

    def __init__(self, root_dir, split="test"):
        self.root_dir = root_dir
        csv_path = os.path.join(root_dir, "splits", f"{split}.csv")
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
            names = sum(rows, [])
        self.paths = [os.path.join(root_dir, "patches", n) for n in names]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        arr = np.load(self.paths[index])        # (C, H, W) float32 [0, 1]
        return torch.from_numpy(arr)
