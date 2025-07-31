import torch
import torch.nn as nn

from timm.layers import trunc_normal_


# CR = 4
def hycass_cr004_spatial0x_n1024(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=4, stages_spatial=0, N=1_024)
def hycass_cr004_spatial1x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=3.96, stages_spatial=1, N=128)
def hycass_cr004_spatial2x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=3.968, stages_spatial=2, N=128)
def hycass_cr004_spatial3x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=3.966, stages_spatial=3, N=128)

# CR = 8
def hycass_cr008_spatial0x_n1024(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=7.7692, stages_spatial=0, N=1_024)
def hycass_cr008_spatial1x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=7.7692, stages_spatial=1, N=128)
def hycass_cr008_spatial2x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=7.7692, stages_spatial=2, N=128)
def hycass_cr008_spatial3x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=7.7692, stages_spatial=3, N=128)

# CR = 16
def hycass_cr016_spatial0x_n1024(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=15.538, stages_spatial=0, N=1_024)
def hycass_cr016_spatial1x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=15.538, stages_spatial=1, N=128)
def hycass_cr016_spatial2x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=15.538, stages_spatial=2, N=128)
def hycass_cr016_spatial3x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=15.538, stages_spatial=3, N=128)

# CR = 32
def hycass_cr032_spatial0x_n1024(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=28.857, stages_spatial=0, N=1_024)
def hycass_cr032_spatial1x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=28.857, stages_spatial=1, N=128)
def hycass_cr032_spatial2x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=28.857, stages_spatial=2, N=128)
def hycass_cr032_spatial3x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=28.857, stages_spatial=3, N=128)

# CR = 50.5
def hycass_cr050_spatial0x_n1024(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=50.5, stages_spatial=0, N=1_024)
def hycass_cr050_spatial1x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=50.5, stages_spatial=1, N=128)
def hycass_cr050_spatial2x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=50.5, stages_spatial=2, N=128)
def hycass_cr050_spatial3x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=50.5, stages_spatial=3, N=128)

# CR = 64 (MLRetSet)
def hycass_cr064_spatial0x_n1024(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=61.5, stages_spatial=0, N=1_024)
def hycass_cr064_spatial1x_n128(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels, img_size, cr_target=61.5, stages_spatial=1, N=128)
def hycass_cr064_spatial2x_n128(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels, img_size, cr_target=61.5, stages_spatial=2, N=128)
def hycass_cr064_spatial3x_n128(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels, img_size, cr_target=61.5, stages_spatial=3, N=128)

# CR = 101.0
def hycass_cr101_spatial0x_n1024(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=101, stages_spatial=0, N=1_024)
def hycass_cr101_spatial1x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=101, stages_spatial=1, N=128)
def hycass_cr101_spatial2x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=101, stages_spatial=2, N=128)
def hycass_cr101_spatial3x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=101, stages_spatial=3, N=128)

# CR = 123 (MLRetSet)
def hycass_cr123_spatial0x_n1024(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=123, stages_spatial=0, N=1_024)
def hycass_cr123_spatial1x_n128(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=123, stages_spatial=1, N=128)
def hycass_cr123_spatial2x_n128(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=123, stages_spatial=2, N=128)
def hycass_cr123_spatial3x_n128(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=123, stages_spatial=3, N=128)

# CR = 185 (MLRetSet)
def hycass_cr185_spatial0x_n1024(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=184.5, stages_spatial=0, N=1_024)
def hycass_cr185_spatial1x_n128(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=184.5, stages_spatial=1, N=128)
def hycass_cr185_spatial2x_n128(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=184.5, stages_spatial=2, N=128)
def hycass_cr185_spatial3x_n128(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=184.5, stages_spatial=3, N=128)

# CR = 202
def hycass_cr202_spatial0x_n1024(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=202, stages_spatial=0, N=1_024)
def hycass_cr202_spatial1x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=202, stages_spatial=1, N=128)
def hycass_cr202_spatial2x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=202, stages_spatial=2, N=128)
def hycass_cr202_spatial3x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=202, stages_spatial=3, N=128)

# CR = 369 (MLRetSet)
def hycass_cr369_spatial0x_n1024(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=369, stages_spatial=0, N=1_024)
def hycass_cr369_spatial1x_n128(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=369, stages_spatial=1, N=128)
def hycass_cr369_spatial2x_n128(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=369, stages_spatial=2, N=128)
def hycass_cr369_spatial3x_n128(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=369, stages_spatial=3, N=128)

# CR = 404
def hycass_cr404_spatial1x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=404, stages_spatial=1, N=128)
def hycass_cr404_spatial2x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=404, stages_spatial=2, N=128)
def hycass_cr404_spatial3x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=404, stages_spatial=3, N=128)

# CR = 738 (MLRetSet)
def hycass_cr738_spatial1x_n128(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=738, stages_spatial=1, N=128)
def hycass_cr738_spatial2x_n128(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=738, stages_spatial=2, N=128)
def hycass_cr738_spatial3x_n128(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=738, stages_spatial=3, N=128)

# CR = 808
def hycass_cr808_spatial1x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=808, stages_spatial=1, N=128)
def hycass_cr808_spatial2x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=808, stages_spatial=2, N=128)
def hycass_cr808_spatial3x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=808, stages_spatial=3, N=128)
def hycass_cr808_spatial4x_n128(src_channels=202, img_size=(128, 128)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=808, stages_spatial=4, N=128)

# CR = 1476 (MLRetSet)
def hycass_cr1476_spatial1x_n128(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=1476, stages_spatial=1, N=128)
def hycass_cr1476_spatial2x_n128(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=1476, stages_spatial=2, N=128)
def hycass_cr1476_spatial3x_n128(src_channels=369, img_size=(96, 96)):
    return AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(src_channels=src_channels, img_size=img_size, cr_target=1476, stages_spatial=3, N=128)


class AdjustableSpatioSpectralHyperspectralImageCompressionNetwork(nn.Module):
    def __init__(self,
                 src_channels=202,
                 img_size=(128,128),
                 cr_target=202,
                 stages_spatial=2,
                 N=128,
                 ):
        super().__init__()

        depths = [2, 4, 6, 2, 2]
        num_heads = [4, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 4.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.2
        norm_layer = nn.LayerNorm
        use_checkpoint= False
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth

        (w, h) = img_size

        self.stages_spatial = stages_spatial
        compression_ratio_spatial = 4 ** stages_spatial

        L = int(src_channels / cr_target * compression_ratio_spatial)
        compression_ratio_spectral = src_channels / L

        self.compression_ratio = compression_ratio_spectral * compression_ratio_spatial
        self.bpppc = 32 / self.compression_ratio

        self.encoder = nn.Sequential(
            # Spectral Encoder Module
            nn.Conv2d(in_channels=src_channels, out_channels=N, kernel_size=1, stride=1, padding="same"),
            nn.LeakyReLU(),
            # Spatial Encoder Module
            nn.Sequential(*[
                nn.Sequential(*[
                    RSTB(
                        dim=N,
                        input_resolution=(h//(2**i),w//(2**i)),
                        depth=depths[i],
                        num_heads=num_heads[i],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                    ),
                    nn.Conv2d(
                        in_channels=N,
                        out_channels=N,
                        kernel_size=3,
                        stride=2,
                        padding=3//2,
                    ),
                    nn.LeakyReLU(),
                ])
                for i in range(stages_spatial)
            ]),
            # CR Adapter Encoder Module
            nn.Conv2d(in_channels=N, out_channels=L, kernel_size=1, stride=1, padding="same"),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            # CR Adapter Decoder Module
            nn.Conv2d(in_channels=L, out_channels=N, kernel_size=1, stride=1, padding="same"),
            nn.LeakyReLU(),
            # Spatial Decoder Module
            nn.Sequential(*[
                nn.Sequential(*[
                    nn.ConvTranspose2d(
                        in_channels=N,
                        out_channels=N,
                        kernel_size=3,
                        stride=2,
                        padding=3//2,
                        output_padding=2-1,
                    ),
                    nn.LeakyReLU(),
                    RSTB(
                        dim=N,
                        input_resolution=(h//(2**i),w//(2**i)),
                        depth=depths[i],
                        num_heads=num_heads[i],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                    ),
                ])
                for i in range(stages_spatial)[::-1]
            ]),
            # Spectral Decoder Module
            nn.Conv2d(in_channels=N, out_channels=src_channels, kernel_size=1, stride=1, padding="same"),
            nn.Sigmoid(),
        )

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.compress(x)
        x = self.decompress(x)
        return x

    def compress(self, x):
        return self.encoder(x)

    def decompress(self, x):
        return self.decoder(x)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


"""
# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


class QReLU(Function):
    """QReLU

    Clamping input with given bit-depth range.
    Suppose that input data presents integer through an integer network
    otherwise any precision of input will simply clamp without rounding
    operation.

    Pre-computed scale with gamma function is used for backward computation.

    More details can be found in
    `"Integer networks for data compression with latent-variable models"
    <https://openreview.net/pdf?id=S1zz2i0cY7>`_,
    by Johannes Ballé, Nick Johnston and David Minnen, ICLR in 2019

    Args:
        input: a tensor data
        bit_depth: source bit-depth (used for clamping)
        beta: a parameter for modeling the gradient during backward computation
    """

    @staticmethod
    def forward(ctx, input, bit_depth, beta):
        # TODO(choih): allow to use adaptive scale instead of
        # pre-computed scale with gamma function
        ctx.alpha = 0.9943258522851727
        ctx.beta = beta
        ctx.max_value = 2**bit_depth - 1
        ctx.save_for_backward(input)

        return input.clamp(min=0, max=ctx.max_value)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        (input,) = ctx.saved_tensors

        grad_input = grad_output.clone()
        grad_sub = (
            torch.exp(
                (-ctx.alpha**ctx.beta)
                * torch.abs(2.0 * input / ctx.max_value - 1) ** ctx.beta
            )
            * grad_output.clone()
        )

        grad_input[input < 0] = grad_sub[input < 0]
        grad_input[input > ctx.max_value] = grad_sub[input > ctx.max_value]

        return grad_input, None, None


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class PatchUnEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, -1, x_size[0], x_size[1])
        return x

    def flops(self):
        flops = 0
        return flops


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         use_checkpoint=use_checkpoint
                                         )

        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed()

    def forward(self, x):
        return self.patch_unembed(self.residual_group(self.patch_embed(x), self.input_resolution), self.input_resolution) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


if __name__ == '__main__':
    import torch

    model = AdjustableSpatioSpectralHyperspectralImageCompressionNetwork()
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
