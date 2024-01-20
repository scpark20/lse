import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Latent(nn.Module):
    def __init__(self, init_log_sigma, const_sigma=False, **kwargs):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.ones(1) * init_log_sigma, requires_grad=not const_sigma)
                
    def forward(self, data, **kwargs):
        # data['z'] : (N, c)
        # data['e'] : (M, c)
        
        N = len(data['z'])
        T = kwargs['temperature'] if 'temperature' in kwargs else 1.0
        z_dim = 1
        
        # (N, 1, z)
        z = data['z'][:, None, :]
        # (1, M, z)
        e = data['e'][None, :, :]
        # (N, M, z)
        distance = (z - e) ** 2
        alpha = -1/(2*torch.exp(self.log_sigma)**2)
        # (M, z)
        loss = T*torch.logsumexp(alpha*distance/T, dim=0)
        # (z,)
        loss = -torch.mean(loss, dim=0) + self.log_sigma - 0.5*np.log(np.e) + np.log(N)
        data['lse_loss'] = torch.mean(loss)
        
        if self.training:
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
        