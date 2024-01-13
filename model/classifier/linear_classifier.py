import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, z_dim, n_class):
        super().__init__()
        self.linear = nn.Linear(z_dim, n_class)
        
    def forward(self, data, **kwargs):
        # z : (b, z)
        # t : (b,)
        
        logit = self.linear(data['z'])
        loss = F.cross_entropy(logit, data['t'])
        data['cross_entropy_loss'] = loss
        
        predict = torch.argmax(logit, dim=1)
        accuracy = torch.sum(predict == data['t']) / len(data['t'])
        data['accuracy'] = accuracy
        
        return data
        