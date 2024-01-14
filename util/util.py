import torch
import librosa
import numpy as np

def save_model_list(save_dir, step, model_list, optimizer_list):
    path = save_dir + 'save_' + str(step)
    models_state_dict = {}
    optimizers_state_dict = {}
    for i, (model, optimizer) in enumerate(zip(model_list, optimizer_list)):
        models_state_dict[i] = model.state_dict()
        optimizers_state_dict[i] = optimizer.state_dict()
    torch.save({'step': step,
                'models_state_dict': models_state_dict,
                'optimizers_state_dict': optimizers_state_dict},
                path)    
    
def save(save_dir, step, model, optimizer):
    path = save_dir + 'save_' + str(step)
    torch.save({'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, 
                path)
    
def load(save_dir, step, model, optimizer):
    path = save_dir + 'save_' + str(step)
    checkpoint = torch.load(path, map_location=torch.device('cpu'))    
    step = checkpoint['step']
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
    return step, model, optimizer

def get_size(model):
    return sum([param.nelement() * param.element_size() for param in model.parameters()]) / 1024**2
