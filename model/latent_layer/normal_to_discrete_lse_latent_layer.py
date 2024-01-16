import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_embeddings(n_embeddings):
    # PyTorch를 사용하여 10개의 점을 등간격으로 샘플링 (수정)
    angles = torch.linspace(0, 2 * torch.pi, n_embeddings)

    # x, y 좌표 계산 (수정)
    x = torch.cos(angles)
    y = torch.sin(angles)
    embeddings = torch.stack([x, y], dim=1) * 4
    return embeddings

class LatentLayer(nn.Module):
    def __init__(self, n_embeddings, z_dim, init_log_sigma, **kwargs):
        super().__init__()
        self.embeddings = nn.Parameter(get_embeddings(n_embeddings), requires_grad=False)
        self.log_sigma = nn.Parameter(torch.ones(1) * init_log_sigma)
                
    def forward(self, data, **kwargs):

        M = kwargs['M']
        sigma = torch.exp(self.log_sigma)
        z = data['z']
        z_dim = z.shape[1]
        N = z.shape[0]
        T = kwargs['temperature']
        n_embeddings = len(self.embeddings)
        
        # (n_embeddings, z)
        mean = (1 - T) * self.embeddings
        # (M//n_embeddings, z)
        eps = torch.randn(M//n_embeddings, z_dim).to(z.device)
        # (n_embeddings, M//n_embeddings, z)
        e = mean.unsqueeze(1) + eps.unsqueeze(0) * T
        # (M, z)
        e = e.reshape(-1, z_dim)
        # (N, M) = sum((N, 1, z) - (1, M, z), dim=2)
        distance = torch.norm(z.unsqueeze(1) - e.unsqueeze(0), dim=2) ** 2
        alpha = -1/(2*sigma**2)
        loss = -torch.mean(torch.logsumexp(alpha*distance, dim=0))
        loss = loss + 0.5*z_dim*(2*self.log_sigma-np.log(np.e)) + np.log(N)
        data['lse_loss'] = loss
        
        return data
        