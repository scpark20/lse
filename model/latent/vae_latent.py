import torch
import torch.nn as nn
import torch.nn.functional as F

class Latent(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
                
    def forward(self, data, **kwargs):
        # z_mean : (b, c)
        # z_logstd : (b, c)
        
        z_mean = data['z_mean']
        z_logvar = data['z_logvar']
        data['kl'] = -0.5 * (1 + z_logvar - z_mean**2 - z_logvar.exp())
        data['kl_loss'] = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean**2 - z_logvar.exp(), dim=1))
        
        return data
        