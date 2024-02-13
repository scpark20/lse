import torch
import torch.nn as nn
import torch.nn.functional as F
from util.mmd_penalty import mmd_penalty

class Latent(nn.Module):
    def __init__(self, opts, **kwargs):
        super().__init__()
        self.opts = opts
                
    def forward(self, data, **kwargs):        
        data['mmd_loss'] = mmd_penalty(data['z'], data['e'], self.opts)
        return data
        