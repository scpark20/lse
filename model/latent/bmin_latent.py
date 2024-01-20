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
        z_dim = 1
        
        # (N, 1, z)
        z = data['z'][:, None, :]
        # (1, M, z)
        e = data['e'][None, :, :]
        # (N, M, z)
        distance = (z - e) ** 2
        # (M, z)
        loss = torch.min(distance, dim=0).values
        # (z,)
        loss = torch.mean(loss, dim=0)
        data['lse_loss'] = torch.mean(loss)

        z_dim = data['z'].shape[1]
        # (N,)
        random_indices = torch.randint(z_dim, (N,))
        # (N, z_dim)
        range_tensor = torch.arange(z_dim).expand(N, z_dim)
        # (N, z_dim)
        zero_tensor = (range_tensor < random_indices.unsqueeze(1)).float().to(data['z'].device)
        # (N, z_dim)
        data['z_copy'] = data['z']
        data['z'] = data['z'] * zero_tensor
        return data
        