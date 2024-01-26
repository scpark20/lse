import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Quantizer(nn.Module):
    def __init__(self, quantize=True, **kwargs):
        super().__init__()
        self.quantize = quantize
                
    def forward(self, data, **kwargs):
        # data['z'] : (N, z)
        # data['e'] : (M, z)
        
        N, z_dim = data['z'].size()
        z = data['z']
        e = data['e']
        T = kwargs['quant_temp'] if 'quant_temp' in kwargs else 0
        
        # (N, M)
        distances = torch.cdist(z.unsqueeze(0), e.unsqueeze(0)).squeeze(0)
        # (N,)
        min_indices = torch.argmin(distances, dim=1)
        data['min_indices'] = min_indices
        # (N, z)
        z_q = torch.index_select(e, 0, min_indices)
        
        data['commit_loss'] = F.mse_loss(data['z'], z_q)
        
        if self.quantize or 'quantize' in kwargs:
            data['ze'] = data['z']
            data['commit_loss'] = F.mse_loss(data['z'], z_q)
            data['z'] = data['z'] + (1-T)*(z_q - data['z']).detach()
            data['zq'] = data['z']
            
        return data