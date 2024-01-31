# https://github.com/sony/sqvae/blob/main/vision/networks/net_64.py

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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

class DecoderVqResnet64(nn.Module):
    def __init__(self, dim_z, cfgs, flg_bn=True):
        super(DecoderVqResnet64, self).__init__()
        # Resblocks
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb-1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        # Convolution layers
        layers_convt = []
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z, 3, stride=1, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z))
        layers_convt.append(nn.ReLU())
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z // 2))
        layers_convt.append(nn.ReLU())
        layers_convt.append(nn.ConvTranspose2d(dim_z // 2, 3, 4, stride=2, padding=1))
        layers_convt.append(nn.Sigmoid())
        self.convt = nn.Sequential(*layers_convt)
        
    def forward(self, z):
        out_res = self.res(z)
        out = self.convt(out_res)

        return out
    
class Decoder(nn.Module):
    def __init__(self, z_dim=512, n_resblocks=6, **kwargs):
        super().__init__()
        from easydict import EasyDict
        cfgs = EasyDict(num_rb=n_resblocks)
        self.decoder = DecoderVqResnet64(z_dim, cfgs)
        
    def forward(self, data, **kwargs):
        # x : (b, c, h, w)
        data['ye'] = self.decoder(data['ze'])
        data['yq'] = self.decoder(data['zq'])
        data['ye_recon_loss'] = F.mse_loss(data['ye'], data['x'])
        data['yq_recon_loss'] = F.mse_loss(data['yq'], data['x'])
        data['recon_loss'] = (data['ye_recon_loss'] + data['yq_recon_loss']) / 2
        return data