import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, size, in_dim, z_dim, h_dims=[32, 64, 128, 256, 512], **kwargs):
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
        self.linear = nn.Linear(h_dims[-1]*size**2, z_dim*2)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        
    def forward(self, data, **kwargs):
        # x : (b, c, h, w)
        
        x = data['x']
        y = self.convs(x)
        y = torch.flatten(y, start_dim=1)
        y = self.linear(y)
        data['z_mean'], data['z_logvar'] = y.split(y.shape[1]//2, dim=1)
        return data
        