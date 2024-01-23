import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Latent(nn.Module):
    def __init__(self, init_log_sigma, const_sigma, **kwargs):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.ones(1) * init_log_sigma, requires_grad=not const_sigma)
                
    def forward(self, data, **kwargs):
        # data['z'] : (N, c, H, W)
        # data['e'] : (M, c)
        
        z_dim = data['z'].shape[1]
        # (NHW, c)
        z = data['z'].permute(0, 2, 3, 1).reshape(-1, z_dim)
        N = len(z)
        T = kwargs['latent_temp'] if 'latent_temp' in kwargs else 1.0
        softmax_temp = kwargs['softmax_temp'] if 'softmax_temp' in kwargs else 1.0
        
        # (NHW, M) = sum((NHW, 1, z) - (1, M, z), dim=2)
        distance = torch.norm(z.unsqueeze(1) - data['e'].unsqueeze(0), dim=2) ** 2
        alpha = -1/(2*torch.exp(self.log_sigma)**2)
        matrix = alpha*distance/T
        data['matrix'] = matrix
        # (NHW, M)
        belong = data['belong'] if 'belong' in data else None
        loss = -torch.mean(T*CustomLogSumExp.apply(matrix, belong, softmax_temp))
        loss = loss + 0.5*z_dim*(2*self.log_sigma-np.log(np.e)) + np.log(N)        
        data['lse_loss'] = loss
        
        return data
    
class CustomLogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, belong=None, temp=1):
        # input : (N, M)
        # belong : (N, M)
        
        ctx.temp = temp
        # (1, M)
        output = torch.logsumexp(input, dim=0, keepdim=True)
        ctx.save_for_backward(input, output, belong)
        return output.squeeze(0)  # output을 반환할 때는 차원을 줄입니다.

    @staticmethod
    def backward(ctx, grad_output):
        temp = ctx.temp
        # (N, M), (1, M), (N, M)
        input, output, belong = ctx.saved_tensors
        # softmax 함수를 사용하여 그래디언트 계산을 수행합니다.
        # (N, M)
        if belong is None:
            softmax_result = torch.exp(input - output)
        else:
            inner_value = belong * input + (1-belong) * (np.log(max(temp, 1e-15)) + input)
            softmax_result = torch.softmax(inner_value, dim=0)
        grad_input = softmax_result * grad_output.unsqueeze(0)
        return grad_input, None, None