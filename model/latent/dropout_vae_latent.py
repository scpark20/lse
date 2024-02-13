import torch
import torch.nn as nn
import torch.nn.functional as F

class Latent(nn.Module):
    def __init__(self, dropout_rate, **kwargs):
        super().__init__()
        self.dropout_rate = dropout_rate
        print('dropout_rate :', self.dropout_rate)
                
    def forward(self, data, **kwargs):
        # z_mean : (b, c)
        # z_logstd : (b, c)
        
        z_mean = data['z_mean']
        z_logvar = data['z_logvar']
        data['kl'] = -0.5 * (1 + z_logvar - z_mean**2 - z_logvar.exp())
        data['kl_loss'] = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean**2 - z_logvar.exp(), dim=1))
        data['z'] = data['z'] * (torch.rand_like(data['z']) < self.dropout_rate)
        
        return data
        