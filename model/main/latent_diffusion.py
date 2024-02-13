import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, latent_diffusion
                ):
        super().__init__()
        self.latent_diffusion = latent_diffusion
        
    def forward(self, data, **kwargs):

        data.update(self.latent_diffusion(data, **kwargs))
        return data