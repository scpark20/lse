import torch
from torch import nn
from torch.nn import functional as F
from util.vae_helpers import HModule, get_1x1, get_3x3, draw_gaussian_diag_samples
from collections import defaultdict
import numpy as np

class Decoder(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.decoder = VDDecoder(H)
        
    def forward(self, data, **kwargs):
        activations = data['activations']
        y, stats = self.decoder(data['x'], activations, **kwargs)
        data['y'] = y
        data['stats'] = stats
        return data
    
    def sample(self, N):
        sample = self.decoder.forward_uncond(N)
        return sample
        

def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers

def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',')
        for ss in s:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping

class Block(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, residual=False, use_3x3=True, zero_last=False):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out
    
def rand_projections(embedding_dim, num_samples=50, device='cpu'):
    """Generates `num_samples` random samples from the latent space's unit sphere.
       Args:
           embedding_dim (int): embedding dimensionality
           num_samples (int): number of random projection samples
       Return:
           torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    unnormd = torch.randn(num_samples, embedding_dim, device=device)
    projections = unnormd.div(torch.norm(unnormd, dim=1, keepdim=True))
    return projections

def sample_binary_outcomes_torch(probabilities, n_samples):
    # Generate random numbers between 0 and 1 of shape (b, n_samples)
    random_numbers = torch.rand(len(probabilities), n_samples)
    
    # Compare random numbers with probabilities to decide the outcome (1 if random_number < probability, else 0)
    outcomes = (random_numbers < probabilities[:, None]).float()
    
    return outcomes
    
class DecBlock(nn.Module):
    def __init__(self, H, res, mixin, n_blocks):
        super().__init__()
        self.base = res
        self.mixin = mixin
        self.H = H
        self.widths = get_width_settings(H.width, H.custom_width_str)
        width = self.widths[res]
        use_3x3 = res > 2
        cond_width = int(width * H.bottleneck_multiple)
        self.zdim = H.zdim
        self.enc = Block(width * 2, cond_width, H.zdim * 2, residual=False, use_3x3=use_3x3)
        self.prior = Block(width, cond_width, H.zdim * 2 + width, residual=False, use_3x3=use_3x3, zero_last=True)
        self.z_proj = get_1x1(H.zdim, width)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.resnet = Block(width, cond_width, width, residual=True, use_3x3=use_3x3)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.z_fn = lambda x: self.z_proj(x)
        self.mask = nn.Parameter(torch.ones(1, H.zdim, res, res)*-1e+1)

    def sample(self, x, acts):
        z, _ = self.enc(torch.cat([x, acts], dim=1)).chunk(2, dim=1)
        feats = self.prior(x)
        _, _, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        x = x + xpp
        b, c, h, w = z.size()
        z_reshaped = z.permute(2, 3, 0, 1).reshape(h*w, b, c)
        e = torch.randn_like(z_reshaped)
        n, b, c = z_reshaped.shape
        projections = rand_projections(c, device=z_reshaped.device)
        projections_expanded = projections.unsqueeze(0).expand(n, *projections.shape)
        z_proj = torch.bmm(z_reshaped, projections_expanded.transpose(1, 2))
        e_proj = torch.bmm(e, projections_expanded.transpose(1, 2))
        z_proj_sorted, _ = torch.sort(z_proj, dim=1)
        e_proj_sorted, _ = torch.sort(e_proj, dim=1)
        # Compute squared distance
        distance = z_proj_sorted - e_proj_sorted
        distance = torch.abs(distance) ** 2
        
        # Mean over 'b'/'m' and 'projection' dimensions to get mean distance per 'n'
        swae = torch.mean(distance, dim=[1, 2])  # (n,)
        
        return z, x, swae

    def sample_uncond(self, x, t=None, lvs=None):
        n, c, h, w = x.shape
        feats = self.prior(x)
        pm, _, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        x = x + xpp
        if lvs is not None:
            z = lvs
        else:
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)
            z = torch.randn_like(pm)
        return z, x

    def get_inputs(self, xs, activations):
        acts = activations[self.base]
        try:
            x = xs[self.base]
        except KeyError:
            x = torch.zeros_like(acts)
        if acts.shape[0] != x.shape[0]:
            x = x.repeat(acts.shape[0], 1, 1, 1)
        return x, acts

    def forward(self, xs, activations, drop_layer, get_latents=False):
        x, acts = self.get_inputs(xs, activations)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x, swae = self.sample(x, acts)
        
        b, c, h, w = z.size()
        
        z_copy = z
        mask = torch.sigmoid(self.mask)
        mask_loss = torch.mean(mask)
        z = z * mask * drop_layer[:, None, None, None]
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        if get_latents:
            return xs, dict(z=z_copy.detach(), swae=swae, mask_loss=mask_loss)
        return xs, dict(swae=swae, mask_loss=mask_loss)

    def forward_uncond(self, xs, t=None, lvs=None):
        try:
            x = xs[self.base]
        except KeyError:
            ref = xs[list(xs.keys())[0]]
            x = torch.zeros(dtype=ref.dtype, size=(ref.shape[0], self.widths[self.base], self.base, self.base), device=ref.device)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x = self.sample_uncond(x, t, lvs=lvs)
        z = z * torch.sigmoid(self.mask)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        return xs
    
class VDDecoder(HModule):

    def build(self):
        H = self.H
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(H.width, H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(H, res, mixin, n_blocks=len(blocks)))
            resos.add(res)
        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.bias_xs = nn.ParameterList([nn.Parameter(torch.zeros(1, self.widths[res], res, res)) for res in self.resolutions if res <= H.no_bias_above])
        self.gain = nn.Parameter(torch.ones(1, H.width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, H.width, 1, 1))
        self.final_fn = lambda x: x * self.gain + self.bias

    def forward(self, x, activations, get_latents=False, **kwargs):
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        # (b,)
        drop_layer = torch.randint(0, len(self.dec_blocks), size=(len(x),)).to(x.device)
        for i, block in enumerate(self.dec_blocks):
            xs, block_stats = block(xs, activations, i < drop_layer, get_latents=get_latents)
            stats.append(block_stats)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size], stats

    def forward_uncond(self, n, t=None, y=None):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for idx, block in enumerate(self.dec_blocks):
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            xs = block.forward_uncond(xs, temp)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]

    def forward_manual_latents(self, n, latents, t=None):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for block, lvs in itertools.zip_longest(self.dec_blocks, latents):
            xs = block.forward_uncond(xs, t, lvs=lvs)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]
