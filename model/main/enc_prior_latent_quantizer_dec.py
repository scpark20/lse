import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, encoder,
                       prior,
                       latent,
                       quantizer,
                       decoder
                ):
        super().__init__()
        self.encoder = encoder
        self.prior = prior
        self.latent = latent
        self.quantizer = quantizer
        self.decoder = decoder
        
    def forward(self, data, **kwargs):
        data.update(self.encoder(data, **kwargs))
        data.update(self.prior(data, **kwargs))
        data.update(self.latent(data, **kwargs))
        data.update(self.quantizer(data, **kwargs))
        data.update(self.decoder(data, **kwargs))
        return data