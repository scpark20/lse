import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
                
    def forward(self, data, **kwargs):

        M = kwargs['M']
        mean = data['z']
        z_dim = mean.shape[1]
        
        # (M, z)
        e = torch.randn(M, z_dim).cuda()    
        # (N, M) = sum((N, 1, z) - (1, M, z), dim=2)
        distance = torch.norm(mean.unsqueeze(1) - e.unsqueeze(0), dim=2) ** 2
        loss = torch.mean(torch.min(distance, dim=0).values)
        data['lse_loss'] = loss
        
        return data
        