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
                
class Encoder(nn.Module):
    def __init__(self, size, in_dim, z_dim, n_blocks, h_dims=[32, 64, 128, 256, 512], **kwargs):
        super().__init__()
        downs = []
        blocks = []
        for h_dim in h_dims:
            size = size // 2
            down = nn.Sequential(nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=2, padding=1),
                                 nn.BatchNorm2d(h_dim),
                                 nn.ReLU())
            downs.append(down)
            block = nn.Sequential(*[ResBlock(h_dim) for _ in range(n_blocks)])
            blocks.append(block)
            in_dim = h_dim
        self.downs = nn.ModuleList(downs)
        self.blocks = nn.ModuleList(blocks)
        self.linear = nn.Linear(h_dims[-1]*size**2, z_dim)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        
    def forward(self, data, **kwargs):
        # x : (b, c, h, w)
        
        x = data['x']
        for down, block in zip(self.downs, self.blocks):
            x = down(x)
            x = block(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        data['z'] = x
        return data
        