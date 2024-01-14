import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LatentLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
                
    def forward(self, data, **kwargs):

        M = kwargs['M']
        log_sigma = torch.rand(1).to(data['z'].device) * 2 - 3
        sigma = torch.exp(log_sigma)
        data['z'] = torch.tanh(data['z'])
        mean = data['z']
        z_dim = mean.shape[1]
        N = mean.shape[0]
        
        # (M, z)
        e = (torch.rand(M, z_dim) * 2 - 1).cuda()
        # (N, M) = sum((N, 1, z) - (1, M, z), dim=2)
        distance = torch.norm(mean.unsqueeze(1) - e.unsqueeze(0), dim=2) ** 2
        alpha = -1/(2*sigma**2)
        loss = -torch.mean(torch.logsumexp(alpha*distance, dim=0))
        loss = loss + 0.5*z_dim*(2*log_sigma-np.log(np.e)) + np.log(N)
        data['lse_loss'] = loss
        
        return data
        