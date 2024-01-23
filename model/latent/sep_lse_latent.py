import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Latent(nn.Module):
    def __init__(self, init_log_sigma, const_sigma, **kwargs):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.ones(1) * init_log_sigma, requires_grad=not const_sigma)
        
    def _pad(self, z, M):
        # z : (N, z)
        
        N = len(z)
        N_padded = int(np.ceil(N/M)*M)
        z = F.pad(z, (0, 0, 0, N_padded-N))
        return z
                
    def forward(self, data, **kwargs):
        # data['z'] : (N, z)
        # data['e'] : (M, z)
        
        T = kwargs['temperature'] if 'temperature' in kwargs else 1.0
        z = data['z']
        e = data['e']
        N, z_dim = z.size()
        M = len(e)
        
        # (M*, z)
        z = self._pad(z, M)
        # (*, M, z)
        z = z.reshape(-1, M, z_dim)
        
        # (*, M_z, M_e) = sum((*, M_z, 1, z) - (1, 1, M_e, z), dim=2)
        distance = torch.norm(z[:, :, None, :] - e[None, None, :, :], dim=3) ** 2
        alpha = -1/(2*torch.exp(self.log_sigma)**2)
        # (*)
        loss = -torch.mean(T*torch.logsumexp(alpha*distance/T, dim=1), dim=1)
        loss = loss + 0.5*z_dim*(2*self.log_sigma-np.log(np.e)) + np.log(M)
        data['lse_loss'] = torch.mean(loss)

        return data