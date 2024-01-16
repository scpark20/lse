import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
                
    def forward(self, data, **kwargs):

        M = kwargs['M']
        data['z'] = torch.tanh(data['z'])
        mean = data['z']
        z_dim = mean.shape[1]
        
        # (M, z)
        e = (torch.rand(M, z_dim) * 2 - 1).cuda()
        # (N, M) = sum((N, 1, z) - (1, M, z), dim=2)
        distance = torch.sum((mean.unsqueeze(1) - e.unsqueeze(0)) ** 2, dim=2)
        loss = torch.mean(torch.min(distance, dim=0).values)
        
        data['lse_loss'] = loss
        
        return data
        