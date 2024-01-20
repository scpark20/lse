import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Latent(nn.Module):
    def __init__(self, init_log_sigma, const_sigma, **kwargs):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.ones(1) * init_log_sigma, requires_grad=not const_sigma)
                
    def forward(self, data, **kwargs):
        # data['z'] : (N, c, H, W)
        # data['e'] : (M, c)
        
        z_dim = data['z'].shape[1]
        # (NHW, c)
        z = data['z'].permute(0, 2, 3, 1).reshape(-1, z_dim)
        N = len(z)
        T = kwargs['latent_temp'] if 'latent_temp' in kwargs else 1.0
        
        # (NHW, M) = sum((NHW, 1, z) - (1, M, z), dim=2)
        distance = torch.norm(z.unsqueeze(1) - data['e'].unsqueeze(0), dim=2) ** 2
        alpha = -1/(2*torch.exp(self.log_sigma)**2)
        loss = -torch.mean(T*torch.logsumexp(alpha*distance/T, dim=0))
        loss = loss + 0.5*z_dim*(2*self.log_sigma-np.log(np.e)) + np.log(N)        
        data['lse_loss'] = loss
        
        # (1,)
        a = -0.5*z_dim*(np.log(2*np.pi)+2*self.log_sigma)
        # (NHW, M)
        b = -0.5/torch.exp(2*self.log_sigma)
        log_likelihood = a[:, None] + b[:, None]*distance
        data['e_new'] = z.T @ F.softmax(log_likelihood, dim=0)

        return data