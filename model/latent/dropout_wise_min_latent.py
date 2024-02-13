import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Latent(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
                
    def forward(self, data, **kwargs):
        # data['z'] : (N, c)
        # data['e'] : (M, c)
        
        N = len(data['z'])
        z_dim = data['z'].shape[1]
        
        # (N, M, z)
        distance = (data['z'].unsqueeze(1) - data['e'].unsqueeze(0)) ** 2
        loss = torch.mean(torch.min(distance.sum(2), dim=0).values)
        data['min_loss'] = loss
        
        loss = torch.mean(torch.mean(torch.min(distance, dim=0).values, dim=0))
        data['wise_min_loss'] = loss
        
        z = data['z']
        # (N,)
        probs = torch.rand(len(z)).to(z.device)
        # (N, zdim)
        dropout = torch.rand(N, z_dim).to(z.device)
        data['z'] = data['z'] * (dropout < probs[:, None])

        return data
        