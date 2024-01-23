import torch
import numpy as np

def generate_circle_data_torch(n_points, radius=1.0):
    # Create an array of angles from 0 to 2*pi
    angles = torch.linspace(0, 2 * torch.pi, steps=n_points)
    
    # Calculate the x and y coordinates
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    
    # Stack the coordinates together into an n x 2 tensor
    points = torch.stack((x, y), dim=1)
    
    return points