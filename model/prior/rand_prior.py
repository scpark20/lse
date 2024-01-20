import torch
import torch.nn as nn
import torch.nn.functional as F

class Prior(nn.Module):
    def __init__(self, n_prior_embeddings, z_dim, **kwargs):
        super().__init__()
        self.prior = nn.Parameter(torch.rand(n_prior_embeddings, z_dim)*2-1, requires_grad=False)
                
    def forward(self, data, **kwargs):

        # (M, z)
        data['e'] = self.sample()
        return data
    
    def sample(self):
        # (M, z)
        samples = self.prior
        return samples