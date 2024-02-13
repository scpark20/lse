import torch
import torch.nn as nn
import torch.nn.functional as F

class Prior(nn.Module):
    def __init__(self, n_zs, subz_dim, **kwargs):
        super().__init__()
        self.n_zs = n_zs
        self.subz_dim = subz_dim
                
    def forward(self, data, **kwargs):

        M = kwargs['M']
        
        # (M, z)
        data['e'] = self.sample(M)
        return data
    
    def sample(self, M):
        # (M, z)
        samples = torch.randn(self.n_zs, M, self.subz_dim).cuda()
        return samples
        
        