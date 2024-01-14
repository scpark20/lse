import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LatentLayer(nn.Module):
    def __init__(self, init_log_sigma, **kwargs):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.ones(1) * init_log_sigma)
        self.log_temperature = nn.Parameter(torch.zeros(1))
                
    def forward(self, data, **kwargs):

        M = kwargs['M']
        sigma = torch.exp(self.log_sigma)
        data['z'] = torch.tanh(data['z'])
        mean = data['z']
        z_dim = mean.shape[1]
        N = mean.shape[0]
        T = torch.exp(self.log_temperature)
        
        # (M, z)
        e = (torch.rand(M, z_dim) * 2 - 1).cuda()
        # (N, M) = sum((N, 1, z) - (1, M, z), dim=2)
        distance = torch.norm(mean.unsqueeze(1) - e.unsqueeze(0), dim=2) ** 2
        alpha = -1/(2*sigma**2)
        loss = -torch.mean(T*torch.logsumexp(alpha*distance/T, dim=0))
        loss = loss + 0.5*z_dim*(2*self.log_sigma-np.log(np.e)) + np.log(N)
        data['lse_loss'] = loss
        
        return data
        