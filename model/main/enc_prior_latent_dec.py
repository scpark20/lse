import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, encoder,
                       prior,
                       latent,
                       decoder
                ):
        super().__init__()
        self.encoder = encoder
        self.prior = prior
        self.latent = latent
        self.decoder = decoder
        
    def forward(self, data, **kwargs):
        data.update(self.encoder(data, **kwargs))
        data.update(self.prior(data, **kwargs))
        data.update(self.latent(data, **kwargs))
        data.update(self.decoder(data, **kwargs))
        return data
    
    def sample(self, z):
        y = self.decoder.sample(z)
        return y