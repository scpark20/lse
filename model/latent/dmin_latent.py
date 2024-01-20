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
        
        # (N, M) = sum((N, 1, z) - (1, M, z), dim=2)
        distance = torch.norm(data['z'].unsqueeze(1) - data['e'].unsqueeze(0), dim=2) ** 2
        loss = torch.mean(torch.min(distance, dim=0).values)
        data['lse_loss'] = loss
        
        loss = torch.mean(torch.min(distance, dim=1).values)
        data['lse_loss'] += loss

        return data