import csv
import numpy as np
import os
import torch

from einops import rearrange
from torch.utils.data import Dataset


class BerlinUrbanGradient(Dataset):
    """
    Dataset:
        Berlin-Urban-Gradient
    Authors:
        Akpona Okujeni
        Sebastian van der Linden
        Patrick Hostert
    Related Paper:
        Berlin-Urban-Gradient dataset 2009 - An EnMAP Preparatory Flight Campaign
        https://gfzpublic.gfz-potsdam.de/rest/items/item_1480927/component/file_1480933/content
    Cite:
        Okujeni, Akpona; van der Linden, Sebastian; Hostert, Patrick (2016): Berlin-Urban-Gradient dataset 2009 - An EnMAP Preparatory Flight Campaign (Datasets). V. 1.2. GFZ Data Services. https://doi.org/10.5880/enmap.2016.008
    Folder Structure:
        - root_dir/
            - patches/
                - 001.npy
                - 002.npy
                - 003.npy
                - ...
            - splits/
                - test.csv
                - train.csv
                - val.csv
                - ...
    """
    def __init__(self, root_dir, split="train", transform=None, random_subsample_factor=None):
        self.root_dir = root_dir

        self.csv_path = os.path.join(self.root_dir, "splits", f"{split}.csv")
        with open(self.csv_path, newline='') as f:
            csv_reader = csv.reader(f)
            csv_data = list(csv_reader)
            self.npy_paths = sum(csv_data, [])
        self.npy_paths = [os.path.join(self.root_dir, "patches", x) for x in self.npy_paths]

        self.transform = transform

        assert random_subsample_factor is None or np.log2(random_subsample_factor) % 1 ==  0
        self.random_subsample_factor = random_subsample_factor

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, index):
        # get full numpy path
        npy_path = self.npy_paths[index]
        # read numpy data
        img = np.load(npy_path)
        # convert numpy array to pytorch tensor
        img = torch.from_numpy(img)
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
    dataset = BerlinUrbanGradient(
        "./datasets/berlin-urban-gradient/",
        split="train",
    )
    for data in dataset:
        print("length:", len(dataset))

        print("shape:", data.shape)
        print("dtype:", data.dtype)
        break
