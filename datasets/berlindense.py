import csv
import numpy as np
import os
import torch
from torch.utils.data import Dataset


class BerlinDense(Dataset):
    """Berlin-Urban-Gradient re-patched with stride=40 (553 patches, 80x80, 111 bands).

    Same source image as BerlinUrbanGradient but with overlapping patches for
    larger training sets. Patches are float32, normalized to [0, 1] per-band.
    """

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.csv_path = os.path.join(root_dir, "splits", f"{split}.csv")
        with open(self.csv_path, newline="") as f:
            rows = list(csv.reader(f))
        self.npy_paths = [
            os.path.join(root_dir, "patches", row[0])
            for row in rows if row
        ]
        self.transform = transform

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, index):
        img = np.load(self.npy_paths[index]).astype(np.float32)
        img = torch.from_numpy(img)
        if self.transform:
            img = self.transform(img)
        return img


if __name__ == "__main__":
    ds = BerlinDense("./datasets/berlin-dense/", split="train")
    print("train size:", len(ds))
    print("patch shape:", ds[0].shape, "dtype:", ds[0].dtype)
