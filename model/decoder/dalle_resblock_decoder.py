import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DALLEDecoder(nn.Module):
    def __init__(self, out_channels, hidden_dims):
        super(DALLEDecoder, self).__init__()
        hidden_dims = hidden_dims[::-1]
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(ResBlock(hidden_dims[i], hidden_dims[i + 1], stride=2))
        layers.append(nn.ConvTranspose2d(hidden_dims[-1], out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.decoder(x)

class Decoder(nn.Module):
    def __init__(self, hidden_dims=[256, 512], z_dim=512, **kwargs):
        super().__init__()
        self.in_conv = nn.Conv2d(z_dim, hidden_dims[-1], kernel_size=1)
        self.decoder = DALLEDecoder(out_channels=3, hidden_dims=hidden_dims)
        
    def forward(self, data, **kwargs):
        # x : (b, c, h, w)
        data['y'] = F.sigmoid(self.decoder(self.in_conv(data['z'])))
        data['recon_loss'] = F.mse_loss(data['y'], data['x'])
        return data
    
