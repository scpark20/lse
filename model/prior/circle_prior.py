import torch
import torch.nn as nn
import torch.nn.functional as F
from util.circle import generate_circle_data_torch

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
        samples = generate_circle_data_torch(M).cuda()
        samples = F.pad(samples, (0, z_dim - samples.shape[1]))
        return samples