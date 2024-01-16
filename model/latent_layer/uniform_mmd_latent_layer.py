import torch
import torch.nn as nn
import torch.nn.functional as F

'''
opts = {'pz_scale': 1,
        'mmd_kernel': 'RBF', # 'IMQ', 'RBF'
        'pz': 'normal', # 'normal', 'sphere', 'uniform'
        'zdim': 2
       }
'''

class LatentLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
                
    def forward(self, data, **kwargs):

        # (N, C)
        data['z'] = torch.tanh(data['z'])
        
        if 'opts' in kwargs:
            # (N, C)
            e = torch.rand_like(data['z'])*2-1
            opts = kwargs['opts']
            data['mmd_loss'] = mmd_penalty(data['z'], e, opts)
        
        return data
        
def mmd_penalty(sample_qz, sample_pz, opts):
    sigma2_p = opts['pz_scale'] ** 2
    kernel = opts['mmd_kernel']
    n = sample_qz.size(0)
    nf = float(n)
    half_size = (n * n - n) // 2

    norms_pz = torch.sum(sample_pz ** 2, dim=1, keepdim=True)
    dotprods_pz = torch.mm(sample_pz, sample_pz.t())
    distances_pz = norms_pz + norms_pz.t() - 2 * dotprods_pz

    norms_qz = torch.sum(sample_qz ** 2, dim=1, keepdim=True)
    dotprods_qz = torch.mm(sample_qz, sample_qz.t())
    distances_qz = norms_qz + norms_qz.t() - 2 * dotprods_qz

    dotprods = torch.mm(sample_qz, sample_pz.t())
    distances = norms_qz + norms_pz.t() - 2 * dotprods

    if kernel == 'RBF':
        sigma2_k, _ = torch.topk(distances.view(-1), half_size, largest=False)
        sigma2_k = sigma2_k[-1] + torch.topk(distances_qz.view(-1), half_size, largest=False)[0][-1]
        res1 = torch.exp(-distances_qz / (2 * sigma2_k))
        res1 = res1 + torch.exp(-distances_pz / (2 * sigma2_k))
        res1 = res1 * (1 - torch.eye(n, device=sample_qz.device))
        res1 = res1.sum() / (nf * nf - nf)
        res2 = torch.exp(-distances / (2 * sigma2_k))
        res2 = res2.sum() * 2. / (nf * nf)
        stat = res1 - res2
    elif kernel == 'IMQ':
        if opts['pz'] == 'normal':
            Cbase = 2 * opts['zdim'] * sigma2_p
        elif opts['pz'] == 'sphere':
            Cbase = 2
        elif opts['pz'] == 'uniform':
            Cbase = opts['zdim']
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = res1 * (1 - torch.eye(n, device=sample_qz.device))
            res1 = res1.sum() / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = res2.sum() * 2. / (nf * nf)
            stat += res1 - res2
    return stat