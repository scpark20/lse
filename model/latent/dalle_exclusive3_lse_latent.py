import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Latent(nn.Module):
    def __init__(self, n_prior_embeddings, **kwargs):
        super().__init__()
        self.norm_factor = nn.Parameter(torch.ones(1, n_prior_embeddings), requires_grad=False)
                
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
            norm_factor = torch.max(belong * distance, dim=0).values.unsqueeze(0)
            self.norm_factor.data = 0.999 * self.norm_factor + (1-0.999) * norm_factor
            norm_factor = torch.clamp(self.norm_factor, 1e-8)
            distance = distance / norm_factor
            distance = torch.where(distance < 1, distance ** (2/distance_p), distance ** (2*distance_p))
            distance = distance * norm_factor
        loss = -torch.mean(torch.logsumexp(-distance, dim=0))
        data['lse_loss'] = loss
        
        return data