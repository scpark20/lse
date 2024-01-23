import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Latent(nn.Module):
    def __init__(self, init_log_sigma, const_sigma, **kwargs):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.ones(1) * init_log_sigma, requires_grad=not const_sigma)
                
    def forward(self, data, **kwargs):
        # data['z'] : (N, z)
        # data['e'] : (M, z)
        
        T = kwargs['temperature'] if 'temperature' in kwargs else 1.0
        z = data['z']
        e = data['e']
        N = len(z)
        z_dim = z.shape[1]
        
        # (z, 1)
        u = torch.randn(z_dim, 1).to(z.device)
        u = u / torch.norm(u, dim=0)
        
        # (N, 1)
        z_proj = z @ u
        # (M, 1)
        e_proj = e @ e
        
        # (N, M) = sum((N, 1, 1) - (1, M, 1), dim=2)
        distance = torch.norm(z_proj.unsqueeze(1) - e_proj.unsqueeze(0), dim=2) ** 2
        alpha = -1/(2*torch.exp(self.log_sigma)**2)
        loss = -torch.mean(T*torch.logsumexp(alpha*distance/T, dim=0))
        loss = loss + 0.5*(2*self.log_sigma-np.log(np.e)) + np.log(N)        
        data['lse_loss'] = loss

        return data