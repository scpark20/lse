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
        data['e'] = (torch.rand(M, z_dim)*2-1).cuda()
        return data
        