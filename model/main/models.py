import torch
import torch.nn as nn
import copy

class Models(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, data, **kwargs):
        data_list = []
        for model in self.models:
            data = copy.deepcopy(data)
            data = model(data, **kwargs)
            data_list.append(data)
        return data_list