import torch
import torch.nn as nn
import torch.nn.functional as F

class Prior(nn.Module):
    def __init__(self, n_latents, n_prior_embeddings, dim_per_latent, **kwargs):
        super().__init__()
        # (L, M, z)
        self.prior = nn.Parameter(torch.randn(n_latents, n_prior_embeddings, dim_per_latent),
                                   requires_grad=False)
                
    def forward(self, data, **kwargs):
        # (L, M, z)
        data['e'] = self.sample()
        return data
    
    def sample(self):
        # (M, z)
        samples = self.prior
        return samples