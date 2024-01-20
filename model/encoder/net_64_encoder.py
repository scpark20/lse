# https://github.com/sony/sqvae/blob/main/vision/networks/net_64.py

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import nn

## Resblocks
class ResBlock(nn.Module):
    def __init__(self, dim, act="relu"):
        super().__init__()
        if act == "relu":
            activation = nn.ReLU()
        elif act == "elu":
            activation = nn.ELU()
        self.block = nn.Sequential(
            activation,
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            activation,
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderVqResnet64(nn.Module):
    def __init__(self, dim_z, cfgs, flg_bn=True, flg_var_q=False):
        super(EncoderVqResnet64, self).__init__()
        self.flg_variance = flg_var_q
        # Convolution layers
        layers_conv = []
        layers_conv.append(nn.Sequential(nn.Conv2d(3, dim_z // 2, 4, stride=2, padding=1)))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z // 2))
        layers_conv.append(nn.ReLU())
        layers_conv.append(nn.Conv2d(dim_z // 2, dim_z, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z))
        layers_conv.append(nn.ReLU())
        layers_conv.append(nn.Conv2d(dim_z, dim_z, 3, stride=1, padding=1))
        self.conv = nn.Sequential(*layers_conv)
        # Resblocks
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb-1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        self.res_m = ResBlock(dim_z)
        if self.flg_variance:
            self.res_v = ResBlock(dim_z)

    def forward(self, x):
        out_conv = self.conv(x)
        out_res = self.res(out_conv)
        mu = self.res_m(out_res)
        if self.flg_variance:
            log_var = self.res_v(out_res)
            return mu, log_var
        else:
            return mu
        
class Encoder(nn.Module):
    def __init__(self, z_dim=512, n_resblocks=6, **kwargs):
        super().__init__()
        from easydict import EasyDict
        cfgs = EasyDict(num_rb=n_resblocks)
        self.encoder = EncoderVqResnet64(z_dim, cfgs)
             
    def forward(self, data, **kwargs):
        # x : (b, c, h, w)
        data['z'] = self.encoder(data['x'])
        return data
   