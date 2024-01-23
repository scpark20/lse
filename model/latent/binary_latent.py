import torch
import torch.nn as nn
import torch.nn.functional as F

class Latent(nn.Module):
    def __init__(self, n_latents, z_dim, **kwargs):
        super().__init__()
        self.L = n_latents
        self.z_dim = z_dim
        self.c_dim = z_dim // n_latents
        # (1, L, 2, z)
        self.e = nn.Parameter(torch.randn(1, self.L, 2, self.c_dim))
                
    def forward(self, data, **kwargs):
        # data['z'] : (N, z, H, W)
        
        T = kwargs['latent_temp'] if 'latent_temp' in kwargs else 1.0
        
        # (1, L, 2, c)
        e = self.e
        z = data['z']
        N, _, H, W = z.size()
        L = self.L
        z_dim = self.z_dim
        c_dim = self.c_dim
        # (NHW, L, 1, c)
        z = z.permute(0, 2, 3, 1).reshape(-1, L, 1, c_dim)
        
        # (NHW, L, 2, 1) = (1, L, 2, 1) - 2*(NHW, L, 2, 1) + (NHW, L, 1, 1)
        distance = (e**2).sum(3, keepdim=True) -\
                   2*e@z.transpose(2, 3) +\
                   (z.transpose(2, 3)**2).sum(2, keepdim=True)
        # (NHW, L, 2, 1)
        ratio = torch.softmax(-distance/T, dim=2)
        data['ratio'] = ratio[:, :, 0, 0]
        # (NHW, L, z)
        zq = torch.sum(self.e * ratio, dim=2)
        # (N, z, H, W)
        zq = zq.reshape(N, H, W, z_dim).permute(0, 3, 1, 2)
        data['z'] = zq
        
        return data