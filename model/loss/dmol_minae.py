import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.vae_helpers import DmolNet

class Loss(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.out_net = DmolNet(H)
        
    def forward(self, data, **kwargs):
        # data['y'] : (B, C, H, W)
        # data['stats']
        # data['x_target']
        
        x = data['x']
        stats = data['stats']
        x_target = data['x_target']
        distortion_per_pixel = F.mse_loss(data['y'][:, :3], x_target.permute(0, 3, 1, 2))
        rate_per_pixel = 0
        for statdict in stats:
            rate_per_pixel = rate_per_pixel + statdict['kl'] * kwargs['kl_weight']
        elbo = distortion_per_pixel + rate_per_pixel
        
        data['elbo'] = elbo
        data['distortion'] = distortion_per_pixel
        data['rate'] = rate_per_pixel
        
        return data
        
    def sample(self, px_z):
        return px_z[:, :3].permute(0, 2, 3, 1).data.cpu().numpy()
        
        