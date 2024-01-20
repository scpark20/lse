import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Latent(nn.Module):
    def __init__(self, n_latents, dim_per_latent, init_log_sigma, const_sigma, **kwargs):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.ones(1) * init_log_sigma, requires_grad=not const_sigma)
        self.n_latents = n_latents # L
        self.dim_per_latent = dim_per_latent # c
                
    def forward(self, data, **kwargs):
        # data['z'] : (N, z, H, W)
        # data['e'] : (L, M, c)
        
        N, z_dim, H, W = data['z'].size()
        # (N, L, c, H, W)
        z = data['z'].reshape(N, self.n_latents, self.dim_per_latent, H, W)
        # (L, NHW, c)
        z = z.permute(1, 0, 3, 4, 2).reshape(self.n_latents, -1, self.dim_per_latent)
        
        # (L, NHW, M) = sum((L, NHW, 1, c) - (L, 1, M, c), dim=2)
        distance = torch.norm(z.unsqueeze(2) - data['e'].unsqueeze(1), dim=3) ** 2
        alpha = -1/(2*torch.exp(self.log_sigma)**2)
        # (L,)
        loss = -torch.mean(torch.logsumexp(alpha*distance, dim=1), dim=1)
        loss = loss + 0.5*z_dim*(2*self.log_sigma-np.log(np.e)) + np.log(N*H*W)
        data['lse_loss'] = torch.mean(loss)

        return data