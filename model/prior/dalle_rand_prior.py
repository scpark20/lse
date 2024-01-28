import torch
import torch.nn as nn
import torch.nn.functional as F

class Prior(nn.Module):
    def __init__(self, n_prior_embeddings, z_dim, prior_mu=0.99, uniform_max=1,
                 z_activation=None, nonlinear=None, noise=0, **kwargs):
        super().__init__()
        self.M = n_prior_embeddings
        self.z_dim = z_dim
        self.prior_sum = nn.Parameter(torch.rand(n_prior_embeddings, z_dim)*uniform_max*2-uniform_max, requires_grad=False)
        self.prior_elem = nn.Parameter(torch.ones(n_prior_embeddings), requires_grad=False)
        self.mu = prior_mu
        self.z_activation = z_activation
        self.nonlinear = None
        if nonlinear is not None:
            if nonlinear == 0:
                self.nonlinear = nn.Sequential(nn.Conv2d(z_dim, z_dim*4, kernel_size=1),
                                               nn.Tanh(),
                                               nn.Conv2d(z_dim*4, z_dim*4, kernel_size=1),
                                               nn.Tanh(),
                                               nn.Conv2d(z_dim*4, z_dim*4, kernel_size=1),
                                               nn.Tanh(),
                                               nn.Conv2d(z_dim*4, z_dim, kernel_size=1))
        self.noise = noise
    
    @property
    def prior(self):
        e = self.prior_sum / self.prior_elem.unsqueeze(1)
        if self.noise > 0:
            e = e + torch.randn_like(e) * self.noise
        return e
        
    def _quantize(self, ze):
        # ze : (N, z)
        # self.prior : (M, z)
        
        # (N, M)
        distance = (ze**2).sum(dim=1, keepdim=True) -\
                   2*ze@self.prior.T +\
                   (self.prior.T**2).sum(0, keepdim=True)
        # (N,), (N,)
        min_distance, zi = torch.min(distance, dim=1)
        return zi
    
    def _update(self, ze, zi, update=True):
        # ze : (N, z)
        # zi : (N,)
        
        with torch.no_grad():
            '''Calculate current centroids of the z embeddings = codebook_sum/codebook_elem'''
            # (N, M)
            zi_onehot = F.one_hot(zi, num_classes=self.M).float()
            
            if update:
                # (M, z) = (M, N) @ (N, z)
                prior_sum_current = zi_onehot.T @ ze
                # (M,)
                prior_elem_current = zi_onehot.sum(0)

                '''Update current centroids parameters'''
                self.prior_sum.data = self.mu*self.prior_sum.data + (1-self.mu)*prior_sum_current.data
                self.prior_elem.data = self.mu*self.prior_elem.data + (1-self.mu)*prior_elem_current.data
            
        return zi_onehot
            
    def forward(self, data, **kwargs):
        # data['z'] : (N, z, H, W)
        
        # (M, z)
        data['e'] = self.sample()
        if self.nonlinear:
            data['z'] = self.nonlinear(data['z'])
        if self.z_activation:
            data['z'] = self.z_activation(data['z'])
        # (NHW, z)
        ze = data['z'].permute(0, 2, 3, 1).reshape(-1, self.z_dim)
        # (NHW,)
        zi = self._quantize(ze)
        zi_onehot = self._update(ze, zi, self.mu > 0)
        # (NHW, M)
        data['belong'] = zi_onehot
        return data
    
    def sample(self):
        # (M, z)
        samples = self.prior
        return samples