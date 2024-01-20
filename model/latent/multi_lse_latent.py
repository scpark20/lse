import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Latent(nn.Module):
    def __init__(self, n_latents, z_dim, init_log_sigma, const_sigma, **kwargs):
        super().__init__()
        self.n_latents = n_latents
        self.z_dim = z_dim
        self.log_sigma = nn.Parameter(torch.ones(1) * init_log_sigma, requires_grad=not const_sigma)
        self.multi_latents_encoder = nn.Linear(z_dim, n_latents*z_dim)
        self.multi_latents_decoder = nn.Linear(n_latents*z_dim, z_dim)
                
    def forward(self, data, **kwargs):
        # data['z'] : (N, z)
        # data['e'] : (L, M, z)
        
        N = len(data['z'])
        T = kwargs['temperature'] if 'temperature' in kwargs else 1.0
        z_dim = self.z_dim
        
        # (N, z)
        z = data['z']
        # (N, L*z)
        z = self.multi_latents_encoder(z)
        # (N, z)
        data['z'] = self.multi_latents_decoder(z)
        # (N, L, z)
        z = z.reshape(-1, self.n_latents, self.z_dim)
        # (L, N, z)
        z = z.transpose(0, 1)
        data['z_multi'] = z
        
        # (L, N, M) = sum((L, N, 1, z) - (L, 1, M, z), dim=2)
        distance = torch.norm(z.unsqueeze(2) - data['e'].unsqueeze(1), dim=3) ** 2
        alpha = -1/(2*torch.exp(self.log_sigma)**2)
        # (L,)
        loss = -torch.mean(T*torch.logsumexp(alpha*distance/T, dim=1), dim=1)
        loss = loss + 0.5*z_dim*(2*self.log_sigma-np.log(np.e)) + np.log(N)        
        data['lse_loss'] = torch.mean(loss)

        return data