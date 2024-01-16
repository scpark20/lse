import torch
import numpy as np

def generate_spiral_data_torch(points_per_class, num_classes=5):
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
        