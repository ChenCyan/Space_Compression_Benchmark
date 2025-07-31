import csv
import os
import torch

import numpy as np
from scipy.io import loadmat

from einops import rearrange
from torch.utils.data import Dataset


class MLRetSet(Dataset):
    """
    MLRetSet - A Multi-Label Airborne Hyperspectral Benchmark Dataset 
    https://doi.org/10.17605/OSF.IO/H2T8U
    from paper: A Novel Semantic Content-Based Retrieval System for Hyperspectral Remote Sensing Imagery
    Folder Structure:
        - root_dir/
            - data/
                - 0001_*.mat
                - 0002_*.mat
                - ...
            - thumbnails/
                - 0001_*.bmp
                - 0002_*.bmp
                - ...
    """
    def __init__(self, root_dir, split="train", transform=None, random_subsample_factor=None):
        self.root_dir = root_dir

        self.csv_path = os.path.join(self.root_dir, "splits", f"{split}.csv")
        with open(self.csv_path, newline='') as f:
            csv_reader = csv.reader(f)
            csv_data = list(csv_reader)
            self.mat_paths = sum(csv_data, [])
        self.mat_paths = [os.path.join(self.root_dir, "data", x) for x in self.mat_paths]

        self.transform = transform

        assert random_subsample_factor is None or np.log2(random_subsample_factor) % 1 ==  0
        self.random_subsample_factor = random_subsample_factor

    def __len__(self):
        return len(self.mat_paths)

    def __getitem__(self, index):
        # get full image path
        file_path = self.mat_paths[index]
        # read image
        mat_dict = loadmat(file_path)
        keys = list(mat_dict)
        img_key = keys[-1]
        img = mat_dict[img_key]
        # convert dtype from float64 (double-precision) to float32 (float-precision)
        img = img.astype(np.float32)
        # convert numpy array to pytorch tensor
        img = torch.from_numpy(img)
        # swap axes (H W C -> C H W)
        img = img.permute(2, 0, 1)
        # apply transformations
        if self.transform:
            img = self.transform(img)
        # pick random pixels:
        if self.random_subsample_factor:
            c, h, w = img.shape

            sample_size = int(h / self.random_subsample_factor) ** 2

            flattened_tensor = img.flatten(1, 2)

            num_elements = flattened_tensor.size(1)

            random_indixes = torch.randperm(num_elements)[:sample_size]

            subsampled_tensor = flattened_tensor[:, random_indixes]

            img = rearrange(subsampled_tensor, 'c (h w) -> c h w',
                h = int(h / self.random_subsample_factor),
                w = int(w / self.random_subsample_factor),
            )
        return img

if __name__ == '__main__':
    import torchvision

    dataset = MLRetSet(
        "./datasets/MLRetSet/",
        split="train",
        transform=torchvision.transforms.CenterCrop(96),
        random_subsample_factor=8,
    )

    print("len:\t", len(dataset))
    
    data_org = dataset[0]

    print("shape:\t", data_org.shape)
    print("dtype:\t", data_org.dtype)
    print("min:\t", data_org.min())
    print("max:\t", data_org.max())
