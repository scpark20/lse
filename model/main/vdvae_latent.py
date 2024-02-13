import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, encoder,
                       decoder,
                       loss,
                ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        
    def forward(self, data, **kwargs):
        data.update(self.encoder(data, **kwargs))
        data.update(self.decoder(data, **kwargs))
        data.update(self.loss(data, **kwargs))
        return data
    
    def get_latent(self, data, **kwargs):
        data.update(self.encoder(data, **kwargs))
        data.update(self.decoder(data, **kwargs))
        return data
        
    def sample(self, N):
        px_z = self.decoder.sample(N)
        sample = self.loss.sample(px_z)
        return sample