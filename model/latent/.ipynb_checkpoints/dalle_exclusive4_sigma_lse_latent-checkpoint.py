import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Latent(nn.Module):
    def __init__(self, init_log_sigma, const_sigma, n_prior_embeddings, **kwargs):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.ones(1) * init_log_sigma, requires_grad=not const_sigma)
        self.max_distance = nn.Parameter(torch.ones(1, n_prior_embeddings), requires_grad=False)
                
    def forward(self, data, **kwargs):
        # data['z'] : (N, c, H, W)
        # data['e'] : (M, c)
        
        z_dim = data['z'].shape[1]
        # (NHW, c)
        z = data['z'].permute(0, 2, 3, 1).reshape(-1, z_dim)
        N = len(z)
        distance_p = kwargs['distance_p'] if 'distance_p' in kwargs else 1
        
        # (NHW, M) = sum((NHW, 1, z) - (1, M, z), dim=2)
        distance = torch.norm(z.unsqueeze(1) - data['e'].unsqueeze(0), dim=2)
        # (NHW, M)
        belong = data['belong'] if 'belong' in data else None
        
        if belong is not None:
            # (1, M)
            max_distance = torch.max(belong * distance, dim=0).values.unsqueeze(0)
            self.max_distance.data = 0.999 * self.max_distance + (1-0.999) * max_distance
            max_distance = torch.clamp(self.max_distance, 1e-8)
            distance = torch.where(distance < max_distance, distance ** (2/distance_p), distance ** (2*distance_p))
        alpha = -1/(2*torch.exp(self.log_sigma)**2)
        loss = -torch.mean(torch.logsumexp(alpha*distance, dim=0))
        loss = loss + z_dim*self.log_sigma
        data['lse_loss'] = loss
        
        return data