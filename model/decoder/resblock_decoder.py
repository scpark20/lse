import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=1),
                                   nn.BatchNorm2d(in_dim),
                                   nn.ReLU(),
                                   nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(in_dim),
                                   nn.ReLU(),
                                   nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(in_dim),
                                   nn.ReLU(),
                                   nn.Conv2d(in_dim, in_dim, kernel_size=1))
        
    def forward(self, x):
        y = x + self.block(x)
        return y

class Decoder(nn.Module):
    def __init__(self, size, out_dim, z_dim, n_blocks, h_dims=[32, 64, 128, 256, 512], **kwargs):
        super().__init__()
        
        size = size // 2 ** len(h_dims)
        self.size = size
        in_dim = h_dims[-1]
        self.in_dim = in_dim
        self.linear = nn.Linear(z_dim, in_dim*size**2)
        
        h_dims = h_dims[:-1]
        ups = []
        blocks = []
        for h_dim in h_dims[::-1]:
            up = nn.Sequential(nn.ConvTranspose2d(in_dim, h_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.BatchNorm2d(h_dim),
                                 nn.ReLU())
            ups.append(up)
            block = nn.Sequential(*[ResBlock(h_dim) for _ in range(n_blocks)])
            blocks.append(block)
            in_dim = h_dim
        self.ups = nn.ModuleList(ups)
        self.blocks = nn.ModuleList(blocks)
        self.out_conv = nn.Sequential(nn.ConvTranspose2d(h_dim, h_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                                      nn.BatchNorm2d(h_dim),
                                      nn.ReLU(),
                                      nn.Conv2d(h_dim, out_dim, kernel_size=3, padding=1),
                                      nn.Tanh())
        
    def forward(self, data, **kwargs):
        # x : (b, c, h, w)
        
        z = data['z']
        y = self.linear(z)
        y = y.reshape(y.shape[0], self.in_dim, self.size, self.size)
        for up, block in zip(self.ups, self.blocks):
            y = up(y)
            y = block(y)
        y = self.out_conv(y)
        data['y'] = y
        data['recon_loss'] = F.mse_loss(data['y'], data['x'])
        return data
    
    def sample(self, z):
        y = self.linear(z)
        y = y.reshape(y.shape[0], self.in_dim, self.size, self.size)
        for up, block in zip(self.ups, self.blocks):
            y = up(y)
            y = block(y)
        y = self.out_conv(y)
        return y