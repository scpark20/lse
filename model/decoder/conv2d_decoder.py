import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, size, out_dim, z_dim, h_dims=[32, 64, 128, 256, 512], **kwargs):
        super().__init__()
        
        size = size // 2 ** len(h_dims)
        self.size = size
        in_dim = h_dims[-1]
        self.in_dim = in_dim
        self.linear = nn.Linear(z_dim, in_dim*size**2)
        
        h_dims = h_dims[:-1]
        convs = []
        for h_dim in h_dims[::-1]:
            conv = nn.Sequential(nn.ConvTranspose2d(in_dim, h_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.BatchNorm2d(h_dim),
                                 nn.LeakyReLU())
            convs.append(conv)
            in_dim = h_dim
        self.convs = nn.Sequential(*convs)
        self.out_conv = nn.Sequential(nn.ConvTranspose2d(h_dim, h_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                                      nn.BatchNorm2d(h_dim),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(h_dim, 3, kernel_size=3, padding=1),
                                      nn.Tanh())
        
    def forward(self, data, **kwargs):
        # x : (b, c, h, w)
        
        z = data['z']
        y = self.linear(z)
        y = y.reshape(y.shape[0], self.in_dim, self.size, self.size)
        y = self.convs(y)
        y = self.out_conv(y)
        data['y'] = y
        data['recon_loss'] = F.mse_loss(data['y'], data['x'])
        return data
    
    def sample(self, z):
        y = self.linear(z)
        y = y.reshape(y.shape[0], self.in_dim, self.size, self.size)
        y = self.convs(y)
        y = self.out_conv(y)
        return y