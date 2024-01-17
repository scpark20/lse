import torch
import torch.nn as nn
import torch.nn.functional as F

class Prior(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
                
    def forward(self, data, **kwargs):

        M = kwargs['M']
        z_dim = data['z'].shape[1]
        
        # (M, z)
        data['e'] = self.sample(M, z_dim)
        return data
        
    def sample(self, M, z_dim):
        # (M, z)
        samples = torch.rand(M, z_dim).cuda() * 2 - 1
        return samples
        
        