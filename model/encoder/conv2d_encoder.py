import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, size, in_dim, z_dim, h_dims=[32, 64, 128, 256, 512], z_activation=None, **kwargs):
        super().__init__()
        convs = []
        for h_dim in h_dims:
            size = size // 2
            conv = nn.Sequential(nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(h_dim),
                                   nn.LeakyReLU())
            convs.append(conv)
            in_dim = h_dim
        self.convs = nn.Sequential(*convs)
        self.linear = nn.Linear(h_dims[-1]*size**2, z_dim)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.activation = z_activation
        
    def forward(self, data, **kwargs):
        # x : (b, c, h, w)
        
        x = data['x']
        y = self.convs(x)
        y = torch.flatten(y, start_dim=1)
        y = self.linear(y)
        if self.activation is not None:
            y = self.activation(y)
        data['z'] = y
        return data
        