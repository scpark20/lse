import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Latent(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
                
    def forward(self, data, **kwargs):
        # data['z'] : (N, c, H, W)
        # data['e'] : (M, c)
        
        z_dim = data['z'].shape[1]
        # (NHW, c)
        z = data['z'].permute(0, 2, 3, 1).reshape(-1, z_dim)
        N = len(z)
        
        # (NHW, M) = sum((NHW, 1, z) - (1, M, z), dim=2)
        distance = torch.norm(z.unsqueeze(1) - data['e'].unsqueeze(0), dim=2) ** 2
        data['matrix'] = distance
        loss = torch.mean(torch.min(distance, dim=0).values)
        data['lse_loss'] = loss
        
        return data