import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LatentLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
                
    def forward(self, data, **kwargs):

        M = kwargs['M']
        data['z'] = data['z']
        mean = data['z']
        z_dim = mean.shape[1]
        
        # (M, z)
        e = generate_spiral_data_torch(M//5, 5)[0].to(mean.device)
        # (N, M) = sum((N, 1, z) - (1, M, z), dim=2)
        distance = torch.sum((mean.unsqueeze(1) - e.unsqueeze(0)) ** 2, dim=2)
        loss = torch.mean(torch.min(distance, dim=0).values)
        
        data['lse_loss'] = loss
        
        return data
    
def generate_spiral_data_torch(points_per_class, num_classes):
    """
    This function generates spiral data with the specified number of points per class
    and the specified number of classes using PyTorch.
    """
    X = []
    y = []
    num_points = points_per_class * num_classes
    radius = torch.linspace(0.0, 1, points_per_class)
    
    for i in range(num_classes):
        theta = torch.linspace(i * 4 * np.pi / num_classes, (i + 1) * 4 * np.pi / num_classes, points_per_class) + torch.randn(points_per_class) * 0.2
        x1 = radius * torch.sin(theta)
        x2 = radius * torch.cos(theta)
        X.append(torch.stack((x1, x2), dim=1))
        y += [i] * points_per_class

    X = torch.cat(X)
    y = torch.tensor(y)
    return X, y
        