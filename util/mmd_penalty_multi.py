import torch

def mmd_penalty_multi(sample_qz, sample_pz, opts):
    sigma2_p = opts['pz_scale'] ** 2
    kernel = opts['mmd_kernel']
    n_instances, n, c = sample_qz.shape
    nf = float(n)
    half_size = (n * n - n) // 2

    # Norms and distances calculation now considers the additional dimension
    norms_pz = torch.sum(sample_pz ** 2, dim=2, keepdim=True)
    dotprods_pz = torch.matmul(sample_pz, sample_pz.transpose(2, 1))
    distances_pz = norms_pz + norms_pz.transpose(2, 1) - 2 * dotprods_pz

    norms_qz = torch.sum(sample_qz ** 2, dim=2, keepdim=True)
    dotprods_qz = torch.matmul(sample_qz, sample_qz.transpose(2, 1))
    distances_qz = norms_qz + norms_qz.transpose(2, 1) - 2 * dotprods_qz

    dotprods = torch.matmul(sample_qz, sample_pz.transpose(2, 1))
    distances = norms_qz + norms_pz.transpose(2, 1) - 2 * dotprods

    stats = torch.zeros(n_instances, device=sample_qz.device)
    if kernel == 'RBF':
        for i in range(n_instances):
            sigma2_k, _ = torch.topk(distances[i].view(-1), half_size, largest=False)
            sigma2_k = sigma2_k[-1] + torch.topk(distances_qz[i].view(-1), half_size, largest=False)[0][-1]
            res1 = torch.exp(-distances_qz[i] / (2 * sigma2_k))
            res1 += torch.exp(-distances_pz[i] / (2 * sigma2_k))
            res1 *= (1 - torch.eye(n, device=sample_qz.device))
            res1 = res1.sum() / (nf * nf - nf)
            res2 = torch.exp(-distances[i] / (2 * sigma2_k))
            res2 = res2.sum() * 2. / (nf * nf)
            stats[i] = res1 - res2
    elif kernel == 'IMQ':
        for i in range(n_instances):
            stat = 0.
            if opts['pz'] == 'normal':
                Cbase = 2 * opts['zdim'] * sigma2_p
            elif opts['pz'] == 'sphere':
                Cbase = 2
            elif opts['pz'] == 'uniform':
                Cbase = opts['zdim']
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                res1 = C / (C + distances_qz[i])
                res1 += C / (C + distances_pz[i])
                res1 *= (1 - torch.eye(n, device=sample_qz.device))
                res1 = res1.sum() / (nf * nf - nf)
                res2 = C / (C + distances[i])
                res2 = res2.sum() * 2. / (nf * nf)
                stat += res1 - res2
            stats[i] = stat
    return stats
