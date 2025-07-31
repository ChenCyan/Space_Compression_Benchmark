from torch import nn


def sscnet_cr004(src_channels=202):
    return SpectralSignalsCompressorNetwork(src_channels, target_compression_ratio=4)


def sscnet_cr008(src_channels=202):
    return SpectralSignalsCompressorNetwork(src_channels, target_compression_ratio=8)


def sscnet_cr016(src_channels=202):
    return SpectralSignalsCompressorNetwork(src_channels, target_compression_ratio=15.8431372549)


def sscnet_cr032(src_channels=202):
    return SpectralSignalsCompressorNetwork(src_channels, target_compression_ratio=32)


def sscnet_cr051(src_channels=202):
    return SpectralSignalsCompressorNetwork(src_channels, target_compression_ratio=50.5)


def sscnet_cr101(src_channels=202):
    return SpectralSignalsCompressorNetwork(src_channels, target_compression_ratio=101)


def sscnet_cr128(src_channels=202):
    return SpectralSignalsCompressorNetwork(src_channels, target_compression_ratio=128)


def sscnet_cr185(src_channels=202):
    return SpectralSignalsCompressorNetwork(src_channels, target_compression_ratio=184.5)


def sscnet_cr202(src_channels=202):
    return SpectralSignalsCompressorNetwork(src_channels, target_compression_ratio=202)


def sscnet_cr269(src_channels=202):
    return SpectralSignalsCompressorNetwork(src_channels, target_compression_ratio=269.3)


def sscnet_cr512(src_channels=202):
    return SpectralSignalsCompressorNetwork(src_channels, target_compression_ratio=512)


def sscnet_cr1024(src_channels=202):
    return SpectralSignalsCompressorNetwork(src_channels, target_compression_ratio=1_024)


class SpectralSignalsCompressorNetwork(nn.Module):
    """
    Title:
        HYPERSPECTRAL DATA COMPRESSION USING FULLY CONVOLUTIONAL AUTOENCODER
    Authors:
        La Grassa, Riccardo and Re, Cristina and Cremonese, Gabriele and Gallo, Ignazio
    Paper:
        https://doi.org/10.3390/rs14102472  
    Cite:
        @article{la2022hyperspectral,
            title={Hyperspectral Data Compression Using Fully Convolutional Autoencoder},
            author={La Grassa, Riccardo and Re, Cristina and Cremonese, Gabriele and Gallo, Ignazio},
            journal={Remote Sensing},
            volume={14},
            number={10},
            pages={2472},
            year={2022},
            publisher={MDPI}
        }
    """

    def __init__(self, src_channels=202, target_compression_ratio=4):
        super(SpectralSignalsCompressorNetwork, self).__init__()

        self.src_channels = src_channels

        self.spatial_downsamplings = 3
        self.spatial_downsampling_factor = 2 ** self.spatial_downsamplings

        self.spectral_downsampling_factor_estimated = target_compression_ratio / self.spatial_downsampling_factor ** 2
        self.latent_channels = int(self.src_channels / self.spectral_downsampling_factor_estimated)
        self.spectral_downsampling_factor = self.src_channels / self.latent_channels

        self.compression_ratio = self.spectral_downsampling_factor * self.spatial_downsampling_factor ** 2
        self.bpppc = 32.0 / self.compression_ratio

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=src_channels,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.PReLU(num_parameters=256),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.PReLU(num_parameters=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.PReLU(num_parameters=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                padding=1,
            ),
            nn.PReLU(num_parameters=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=512,
                out_channels=self.latent_channels,
                kernel_size=3,
                padding=1
            ),
            nn.PReLU(num_parameters=self.latent_channels),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.latent_channels,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PReLU(num_parameters=512),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=2,
                stride=2,
            ),
            nn.PReLU(num_parameters=256),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=2,
                stride=2,
            ),
            nn.PReLU(num_parameters=256),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=2,
                stride=2,
            ),
            nn.PReLU(num_parameters=256),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=src_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.compress(x)
        x_hat = self.decompress(y)
        return x_hat

    def compress(self, x):
        y = self.encoder(x)
        return y

    def decompress(self, y):
        x_hat = self.decoder(y)
        return x_hat

    @classmethod
    def from_state_dict(cls, state_dict):
        net = cls()
        net.load_state_dict(state_dict)
        return net


if __name__ == '__main__':
    import torch

    model = SpectralSignalsCompressorNetwork()
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
