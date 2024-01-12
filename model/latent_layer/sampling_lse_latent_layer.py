import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.z_logvar = nn.Parameter(torch.zeros(1, 1))
                
    def forward(self, data, **kwargs):

        M = kwargs['M']
        # (N, z)
        z_mean = data['z_mean']
        data['z'] = z_mean + torch.exp(0.5*self.z_logvar) * torch.randn_like(z_mean)
        z_dim = z_mean.shape[1]
        
        # (M, z)
        e = torch.randn(M, z_dim).cuda()
        # (N, M) = sum((N, 1, z) - (1, M, z), dim=2)
        distance = torch.norm(z_mean.unsqueeze(1) - e.unsqueeze(0), dim=2) ** 2
        # (1, 1)
        alpha = -1 / 2 * torch.exp(self.z_logvar)
        loss = torch.mean(-torch.logsumexp(alpha * distance, dim=0)) + 0.5*z_dim*self.z_logvar
        data['lse_loss'] = loss
        
        return data