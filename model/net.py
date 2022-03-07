import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import UpSample
from model.basenet import BaseNet
from model.layer import (SpectralNorm,)
from functools import partial
import numpy as np

# generative multi-column convolutional neural net
class GMCNN(BaseNet):
    def __init__(
        self,
        in_channels,
        out_channels,
        cnum=32,
        act=F.elu,
        norm=F.instance_norm,
        using_norm=False,
    ):
        super(GMCNN, self).__init__()
        self.act = act
        self.using_norm = using_norm
        if using_norm is True:
            self.norm = norm
        else:
            self.norm = None
        ch = cnum

        # network structure
        self.EB1 = []
        self.EB2 = []
        self.EB3 = []
        self.decoding_layers = []

        self.EB1_pad_rec = []
        self.EB2_pad_rec = []
        self.EB3_pad_rec = []

        self.EB1.append(nn.Conv3d(in_channels, ch, kernel_size=7, stride=1))

        self.EB1.append(nn.Conv3d(ch, ch * 2, kernel_size=7, stride=2))
        self.EB1.append(nn.Conv3d(ch * 2, ch * 2, kernel_size=7, stride=1))

        self.EB1.append(nn.Conv3d(ch * 2, ch * 4, kernel_size=7, stride=2))
        self.EB1.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=7, stride=1))

        self.EB1.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=2))
        self.EB1.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=4))
        self.EB1.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=8))
        self.EB1.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=7, stride=1, dilation=16))

        self.EB1.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=7, stride=1))
        self.EB1.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=7, stride=1))

        self.EB1.append(
            UpSample(
                spatial_dims=3, in_channels=ch * 4, out_channels=ch * 4, scale_factor=4
            )
        )
        # self.EB1.append(PureUpsampling(scale=4))

        self.EB1_pad_rec = [3, 3, 3, 3, 3, 3, 6, 12, 24, 48, 3, 3, 0]

        self.EB2.append(nn.Conv3d(in_channels, ch, kernel_size=5, stride=1))

        self.EB2.append(nn.Conv3d(ch, ch * 2, kernel_size=5, stride=2))
        self.EB2.append(nn.Conv3d(ch * 2, ch * 2, kernel_size=5, stride=1))

        self.EB2.append(nn.Conv3d(ch * 2, ch * 4, kernel_size=5, stride=2))
        self.EB2.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=5, stride=1))

        self.EB2.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=2))
        self.EB2.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=4))
        self.EB2.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=8))
        self.EB2.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=5, stride=1, dilation=16))

        self.EB2.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=5, stride=1))

        self.EB2.append(
            UpSample(
                spatial_dims=3, in_channels=ch * 4, out_channels=ch * 4, scale_factor=2
            )
        )
        self.EB2.append(nn.Conv3d(ch * 4, ch * 2, kernel_size=5, stride=1))
        self.EB2.append(nn.Conv3d(ch * 2, ch * 2, kernel_size=5, stride=1))
        self.EB2.append(
            UpSample(
                spatial_dims=3, in_channels=ch * 2, out_channels=ch * 2, scale_factor=2
            )
        )
        self.EB2_pad_rec = [2, 2, 2, 2, 2, 2, 4, 8, 16, 32, 2, 2, 0, 2, 2, 0]

        self.EB3.append(nn.Conv3d(in_channels, ch, kernel_size=3, stride=1))

        self.EB3.append(nn.Conv3d(ch, ch * 2, kernel_size=3, stride=2))
        self.EB3.append(nn.Conv3d(ch * 2, ch * 2, kernel_size=3, stride=1))

        self.EB3.append(nn.Conv3d(ch * 2, ch * 4, kernel_size=3, stride=2))
        self.EB3.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=3, stride=1))

        self.EB3.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=2))
        self.EB3.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=4))
        self.EB3.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=8))
        self.EB3.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=3, stride=1, dilation=16))

        self.EB3.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv3d(ch * 4, ch * 4, kernel_size=3, stride=1))

        self.EB3.append(
            UpSample(
                spatial_dims=3, in_channels=ch * 4, out_channels=ch * 4, scale_factor=2
            )
        )
        self.EB3.append(nn.Conv3d(ch * 4, ch * 2, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv3d(ch * 2, ch * 2, kernel_size=3, stride=1))
        self.EB3.append(
            UpSample(
                spatial_dims=3, in_channels=ch * 2, out_channels=ch * 2, scale_factor=2
            )
        )
        self.EB3.append(nn.Conv3d(ch * 2, ch, kernel_size=3, stride=1))
        self.EB3.append(nn.Conv3d(ch, ch, kernel_size=3, stride=1))

        self.EB3_pad_rec = [1, 1, 1, 1, 1, 1, 2, 4, 8, 16, 1, 1, 0, 1, 1, 0, 1, 1]

        self.decoding_layers.append(nn.Conv3d(ch * 7, ch // 2, kernel_size=3, stride=1))
        self.decoding_layers.append(
            nn.Conv3d(ch // 2, out_channels, kernel_size=3, stride=1)
        )

        self.decoding_pad_rec = [1, 1]

        self.EB1 = nn.ModuleList(self.EB1)
        self.EB2 = nn.ModuleList(self.EB2)
        self.EB3 = nn.ModuleList(self.EB3)
        self.decoding_layers = nn.ModuleList(self.decoding_layers)

        required_padding = list(
            set(
                self.EB1_pad_rec
                + self.EB2_pad_rec
                + self.EB3_pad_rec
                + self.decoding_pad_rec
            )
        )
        self.pad_list = [partial(F.pad, pad=tuple([i]*6)) for i in required_padding]
        self.pads = {pad: self.pad_list[i] for i, pad in enumerate(required_padding)}

        # # padding operations
        # padlen = 49
        # self.pads = [0] * padlen
        # for i in range(padlen):
        #     self.pads[i] = nn.ReflectionPad3d(i)
        # self.pads = nn.ModuleList(self.pads)

        # self.pad_list = nn.ModuleList(self.pad_list)
        # self.pads = lambda p: self.pad_list[self.pad_map[p]]

    def __pad_hack(self, x, pad_idx):
        if pad_idx >= x.shape[2]:
            f = 1
            while pad_idx // f >= x.shape[2]:
                f *= 2
            pad_idx = pad_idx // f
            for _ in range(f - 1):
                x = self.pads[pad_idx](x)
        return self.pads[pad_idx](x)

    def forward(self, x):
        x1, x2, x3 = x, x, x

        for i, layer in enumerate(self.EB1):
            pad_idx = self.EB1_pad_rec[i]
            x1 = self.__pad_hack(x1, pad_idx)
            x1 = layer(x1)
            if self.using_norm:
                x1 = self.norm(x1)
            if pad_idx != 0:
                x1 = self.act(x1)

        for i, layer in enumerate(self.EB2):
            pad_idx = self.EB2_pad_rec[i]
            x2 = self.__pad_hack(x2, pad_idx)
            x2 = layer(x2)
            if self.using_norm:
                x2 = self.norm(x2)
            if pad_idx != 0:
                x2 = self.act(x2)

        for i, layer in enumerate(self.EB3):
            pad_idx = self.EB3_pad_rec[i]
            x3 = self.__pad_hack(x3, pad_idx)
            x3 = layer(x3)
            if self.using_norm:
                x3 = self.norm(x3)
            if pad_idx != 0:
                x3 = self.act(x3)

        x_d = torch.cat((x1, x2, x3), 1)
        x_d = self.act(
            self.decoding_layers[0](self.pads[self.decoding_pad_rec[0]](x_d))
        )
        x_d = self.decoding_layers[1](self.pads[self.decoding_pad_rec[1]](x_d))
        x_out = torch.clamp(x_d, -1, 1) # NOTE: different dynamic range?
        return x_out

# return one dimensional output indicating the probability of realness or fakeness
class Discriminator(BaseNet):
    def __init__(
        self,
        in_channels,
        cnum=32,
        fc_channels=8 * 8 * 32 * 4,
        act=F.elu,
        norm=None,
        spectral_norm=True,
    ):
        super(Discriminator, self).__init__()
        self.act = act
        self.norm = norm
        # self.embedding = None
        # self.logit = None

        ch = cnum
        self.layers = []
        if spectral_norm:
            self.layers.append(
                SpectralNorm(
                    nn.Conv3d(in_channels, ch, kernel_size=5, padding=2, stride=2)
                )
            )
            self.layers.append(
                SpectralNorm(nn.Conv3d(ch, ch * 2, kernel_size=5, padding=2, stride=2))
            )
            self.layers.append(
                SpectralNorm(
                    nn.Conv3d(ch * 2, ch * 4, kernel_size=5, padding=2, stride=2)
                )
            )
            self.layers.append(
                SpectralNorm(
                    nn.Conv3d(ch * 4, ch * 4, kernel_size=5, padding=2, stride=2)
                )
            )
            self.layers.append(SpectralNorm(nn.Linear(fc_channels, 1)))
        else:
            self.layers.append(
                nn.Conv3d(in_channels, ch, kernel_size=5, padding=2, stride=2)
            )
            self.layers.append(
                nn.Conv3d(ch, ch * 2, kernel_size=5, padding=2, stride=2)
            )
            self.layers.append(
                nn.Conv3d(ch * 2, ch * 4, kernel_size=5, padding=2, stride=2)
            )
            self.layers.append(
                nn.Conv3d(ch * 4, ch * 4, kernel_size=5, padding=2, stride=2)
            )
            self.layers.append(nn.Linear(fc_channels, 1))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for _, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.norm is not None:
                x = self.norm(x)
            x = self.act(x)
        embedding = x.view(x.size(0), -1)
        logit = self.layers[-1](embedding)
        return logit

class GlobalLocalDiscriminator(BaseNet):
    def __init__(
        self,
        in_channels,
        cnum=32,
        g_fc_channels=16 * 16 * 32 * 4,
        l_fc_channels=8 * 8 * 32 * 4,
        act=F.elu,
        norm=None,
        spectral_norm=True,
    ):
        super(GlobalLocalDiscriminator, self).__init__()
        self.act = act
        self.norm = norm

        self.global_discriminator = Discriminator(
            in_channels=in_channels,
            fc_channels=g_fc_channels,
            cnum=cnum,
            act=act,
            norm=norm,
            spectral_norm=spectral_norm,
        )
        self.local_discriminator = Discriminator(
            in_channels=in_channels,
            fc_channels=l_fc_channels,
            cnum=cnum,
            act=act,
            norm=norm,
            spectral_norm=spectral_norm,
        )

    def forward(self, x_g, x_l):
        x_global = self.global_discriminator(x_g)
        x_local = self.local_discriminator(x_l)
        return x_global, x_local
