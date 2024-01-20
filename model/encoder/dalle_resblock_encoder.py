import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DALLEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dims):
        super().__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(ResBlock(in_channels, h_dim, stride=2))
            in_channels = h_dim
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.encoder(x)

class Encoder(nn.Module):
    def __init__(self, hidden_dims=[256, 512], z_dim=512, **kwargs):
        super().__init__()
        self.encoder = DALLEEncoder(in_channels=3, hidden_dims=hidden_dims)
        self.out_conv = nn.Conv2d(hidden_dims[-1], z_dim, kernel_size=1)
             
    def forward(self, data, **kwargs):
        # x : (b, c, h, w)
        data['z'] = self.out_conv(self.encoder(data['x']))
        return data
   