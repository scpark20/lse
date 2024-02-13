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
        
        distortion_per_pixel = self.out_net.nll(data['y'], x_target)
        rate_per_pixel = 0
        mask_loss = 0
        ndims = np.prod(x.shape[1:])
        for statdict in stats:
            rate_per_pixel += statdict['swae'].mean() if statdict['swae'] is not None else 0
            mask_loss += statdict['mask_loss'] * kwargs['mask_loss_weight'] if statdict['mask_loss'] is not None else 0
        elbo = distortion_per_pixel.mean() + rate_per_pixel + mask_loss
        print('mask loss :', mask_loss)
        
        data['elbo'] = elbo
        data['distortion'] = distortion_per_pixel.mean()
        data['rate'] = rate_per_pixel
        
        return data
        
    def sample(self, px_z):
        return self.out_net.sample(px_z)
        
        