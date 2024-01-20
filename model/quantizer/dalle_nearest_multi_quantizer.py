import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Quantizer(nn.Module):
    def __init__(self, n_latents, dim_per_latent, **kwargs):
        super().__init__()
        self.n_latents = n_latents # L
        self.dim_per_latent = dim_per_latent # c
                
    def forward(self, data, **kwargs):
        # data['z'] : (N, z, H, W)
        # data['e'] : (L, M, c)
        
        L = self.n_latents
        c = self.dim_per_latent
        # (N, z, H, W)
        z = data['z']
        N, z_dim, H, W = z.size()
        # (N, L, c, H, W)
        z = z.reshape(N, L, c, H, W)
        # (L, NHW, c)
        z = z.permute(1, 0, 3, 4, 2).reshape(L, -1, c)
        # (L, M, c)
        e = data['e']
        
        # (L, NHW, M)
        distances = torch.cdist(z, e)
        # (L, NHW)
        min_indices = torch.argmin(distances, dim=2)
        data['min_indices'] = min_indices.reshape(L, N, H, W)
        z_q = []
        for l in range(L):
            # (NHW, c)
            z_q_ = torch.index_select(e[l], 0, min_indices[l])
            z_q.append(z_q_)
        # (L, NHW, c)
        z_q = torch.stack(z_q, dim=0)
        # (N, z, H, W)
        z_q = z_q.transpose(1, 2).reshape(z_dim, N, H, W).transpose(0, 1)
        
        data['z'] = data['z'] + (z_q - data['z']).detach()
        
        return data