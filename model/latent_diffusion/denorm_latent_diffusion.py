import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from util.vae_helpers import get_1x1, get_3x3

class LatentDiffusion(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.backbones = nn.ModuleList()
        blocks = parse_layer_string(H.dec_blocks)
        for idx, (res, mixin) in enumerate(blocks):
            block = Block(H.zdim, H.diff_middle_width, H.zdim,
                          H.diff_residual, True, True, H.zdim)
            self.backbones.append(block)
        
    def forward(self, data, **kwargs):
        # data['stats'] : [{'pm','pv','z'}, ...]
        
        losses = 0
        for stat, backbone in zip(data['stats'], self.backbones):
            z0 = stat['z']
            mean = stat['pm']
            std = torch.exp(stat['pv'])
            
            z0_norm = (z0 - mean) / std
            e_norm = torch.randn_like(z0_norm)
            e = e_norm * std + mean
            t = torch.randint(0, self.H.scheduler.config.num_train_timesteps, size=(len(z0),)).to(z0.device)
            zt_norm = self.H.scheduler.add_noise(z0_norm, e_norm, t)
            zt = zt_norm * std + mean
            pred = backbone(zt, t)
            if self.H.scheduler.config.prediction_type == 'epsilon':
                loss = F.mse_loss(pred, e)
            elif self.H.scheduler.config.prediction_type == 'sample':
                loss = F.mse_loss(pred, z0)
            losses += loss
        
        data['diffusion_loss'] = losses / len(self.backbones)
        return data
            
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers

class Block(nn.Module):
    def __init__(self, in_width, middle_width, out_width, residual=False, use_3x3=True, zero_last=False, time_width=0):
        super().__init__()
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)
        self.time_width = time_width
        if time_width > 0:
            self.time_embed = get_1x1(time_width, middle_width)

    def forward(self, x, t=None):
        if t is None:
            t = 0
        else:
            t = timestep_embedding(t, self.time_width)[:, :, None, None]
            t = self.time_embed(t)
            
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat) + t)
        xhat = self.c3(F.gelu(xhat) + t)
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        return out
