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
    
def load_model_list(load_dir, step, model_list, optimizer_list):
    path = load_dir + 'save_' + str(step)
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    # warm start
    for i, (model, optimizer) in enumerate(zip(model_list, optimizer_list)):
        model.load_state_dict(checkpoint['models_state_dict'][i], strict=True)
        optimizer.load_state_dict(checkpoint['optimizers_state_dict'][i])
    step = checkpoint['step']
    return step, model_list, optimizer_list
    
def load(save_dir, step, model, optimizer):
    path = save_dir + 'save_' + str(step)
    checkpoint = torch.load(path, map_location=torch.device('cpu'))    
    step = checkpoint['step']
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
    return step, model, optimizer

def get_size(model):
    return sum([param.nelement() * param.element_size() for param in model.parameters()]) / 1024**2
