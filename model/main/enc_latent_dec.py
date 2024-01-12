import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, encoder,
                       latent_layer,
                       decoder
                ):
        super().__init__()
        self.encoder = encoder
        self.latent_layer = latent_layer
        self.decoder = decoder
        
    def forward(self, data, **kwargs):

        data.update(self.encoder(data, **kwargs))
        data.update(self.latent_layer(data, **kwargs))
        data.update(self.decoder(data, **kwargs))
        return data
    
    def sample(self, z):
        y = self.decoder.sample(z)
        return y