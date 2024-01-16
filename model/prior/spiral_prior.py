import torch
import torch.nn as nn
import torch.nn.functional as F
from util.spiral import generate_spiral_data_torch

class Prior(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
                
    def forward(self, data, **kwargs):

        M = kwargs['M']
        z_dim = data['z'].shape[1]
        
        # (M, z)
        data['e'] = generate_spiral_data_torch(M//5+1, 5)[0]
        data['e'] = data['e'][torch.randperm(len(data['e']))]
        data['e'] = data['e'][:M].to(data['z'].device)
        return data
        