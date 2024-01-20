import torch
import numpy as np

def get_nll(p_samples, q_samples, log_sigma, temperature=1.0):
    # p_samples : (M, z)
    # q_samples : (N, z)
    M = len(p_samples)
    N = len(q_samples)
    z_dim = p_samples.shape[1]
    T = temperature
    
    # (N, M)
    distance = torch.norm(q_samples.unsqueeze(1) - p_samples.unsqueeze(0), dim=2) ** 2
    alpha = -1/(2*np.exp(log_sigma)**2)
    nll = -torch.mean(T*torch.logsumexp(alpha*distance/T, dim=0))
    nll = nll + 0.5*z_dim*(2*log_sigma-np.log(np.e)) + np.log(N)
    
    return nll.item()

def get_optimum_log_sigma(p_samples1, p_samples2, min_log_sigma=-3, max_log_sigma=3, temperature=1.0):
    log_sigmas = np.linspace(min_log_sigma, max_log_sigma, 100)
    nlls = np.array([get_nll(p_samples1, p_samples2, log_sigma, temperature) for log_sigma in log_sigmas])
    return log_sigmas[np.argmin(nlls)]

def get_cross_nll(p_samples, q_samples, log_sigma):
    nll_p2q = get_nll(p_samples, q_samples, log_sigma)
    nll_q2p = get_nll(q_samples, p_samples, log_sigma)
    return (nll_p2q + nll_q2p) / 2
    