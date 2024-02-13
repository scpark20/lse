import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Latent(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
                
    def forward(self, data, **kwargs):
        # z : (N, c)
        # data['e'] : (M, c)
        
        z = data['z_mean']
        logits = data['z_logvar']
        probs = torch.softmax(logits, dim=0)
        data['probs'] = probs
        
        N = len(z)
        z_dim = z.shape[1]
        
        # (50, z)
        projections = rand_projections(z_dim, device=z.device)
        # (N, 50)
        z_proj = z @ projections.T
        # (M, 50)
        e_proj = data['e'] @ projections.T
        
        distance = torch.sort(z_proj.T, dim=1)[0] - torch.sort(e_proj.T, dim=1)[0]
        distance = torch.abs(distance) ** 2
        data['swae_loss'] = torch.mean(distance)
        
        data['z'] = z * probs
        data['bottleneck_loss'] = torch.mean(torch.mean(-probs * torch.log(probs + 1e-8) * kwargs['bottleneck_weight'], dim=1))

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
        