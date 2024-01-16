import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Latent(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
                
    def forward(self, data, **kwargs):
        # data['z'] : (N, c)
        # data['e'] : (M, c)
        
        N = len(data['z'])
        z_dim = data['z'].shape[1]
        
        # (50, z)
        projections = rand_projections(z_dim, device=data['z'].device)
        # (N, 50)
        z_proj = data['z'] @ projections.T
        # (M, 50)
        e_proj = data['e'] @ projections.T
        
        distance = torch.sort(z_proj.T, dim=1)[0] - torch.sort(e_proj.T, dim=1)[0]
        distance = torch.abs(distance) ** 2
        data['swae_loss'] = torch.mean(distance)

        return data
    
def rand_projections(embedding_dim, num_samples=50, device='cpu'):
    """This function generates `num_samples` random samples from the latent space's unit sphere.
        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples
        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    unnormd = torch.randn(num_samples, embedding_dim, device=device)

    projections = unnormd.div( torch.norm(unnormd,dim=1,keepdim=True) )
    return projections
        