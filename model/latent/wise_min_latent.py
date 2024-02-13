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
        
        # (N, M, z)
        distance = (data['z'].unsqueeze(1) - data['e'].unsqueeze(0)) ** 2
        loss = torch.mean(torch.min(distance.sum(2), dim=0).values)
        data['min_loss'] = loss
        
        loss = torch.mean(torch.mean(torch.min(distance, dim=0).values, dim=0))
        data['wise_min_loss'] = loss

        return data
        