import glymur
import numpy as np
import os
import torch

from torch import nn


class JPEG2000(nn.Module):
    def __init__(
            self,
            src_channels=202,
            target_compression_ratio=4,
            rescale=False,
        ):
        super().__init__()

        self.rescale = rescale

        self.compression_ratio = target_compression_ratio
        self.compression_ratio_jp2k = target_compression_ratio // 2
        self.bpppc = 32 / self.compression_ratio

        self.jp2_path = "/tmp/tmp.jp2"

    def compress(self, x):
        return x

    def decompress(self, y):
        return y
    
    def forward(self, x):
        assert x.shape[0] == 1, "batch dimension must be 1"

        x = x.squeeze(0)
        if self.rescale:
            x = 10_000 * x
        x = torch.Tensor.numpy(x)
        x = x.astype(np.uint16)
        x = np.moveaxis(x, 0, 2)

        jp2k = glymur.Jp2k(self.jp2_path, x, cratios=[self.compression_ratio_jp2k])

        x_hat = jp2k[:]
        x_hat = np.moveaxis(x_hat, 2, 0)
        x_hat = x_hat.astype(np.float32)
        if self.rescale:
            x_hat = x_hat / 10_000
        
        x_hat = torch.from_numpy(x_hat)
        x_hat = x_hat.unsqueeze(0)
        return x_hat


if __name__ == '__main__':
    import torch

    model = JPEG2000()
    print(model)

    in_tensor = torch.randn(1, 202, 128, 128)
    print("in shape:\t\t", in_tensor.shape)

    latent_tensor = model.compress(in_tensor)
    print("latent shape:\t\t", latent_tensor.shape)
    
    out_tensor = model(in_tensor)
    print("out shape:\t\t", out_tensor.shape)

    print("in shape = out shape:\t", out_tensor.shape == in_tensor.shape)

    print("real bpppc:\t\t", 32 * torch.numel(latent_tensor) / torch.numel(in_tensor))
    print("model parameter bpppc:\t", model.bpppc)
