from .logging import *
from .dataset import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def get_mask_from_lengths(lengths, max_len=None):
    '''
    lengths: [B, ]
    return a float type mask having below shape
    [[1.0, 1.0, 1.0, 1.0, 0.0],
     [1.0, 1.0, 0.0, 0.0, 0.0],
     [1.0, 1.0, 1.0, 1.0, 1.0]]
    '''
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(max_len, device=lengths.device)
    mask = (ids < lengths.unsqueeze(-1)).float()
    return mask


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def MMD(preds, targets, sigma=10.0):
    # preds, targets: (B, T, D)
    n=preds.size(0)
    x, y = preds.reshape(n, -1), targets.reshape(n, -1)
    
    Kxx = torch.exp(-torch.pow(torch.cdist(x, x), 2) / (2 * sigma ** 2)) - torch.eye(n).to(x.device)
    Kxx = torch.sum(Kxx) / (n*(n-1))
    
    Kxy = torch.exp(-torch.pow(torch.cdist(x, y), 2) / (2 * sigma ** 2))
    Kxy = torch.sum(Kxy) / (n*n)
    
    Kyy = torch.exp(-torch.pow(torch.cdist(y, y), 2) / (2 * sigma ** 2)) - torch.eye(n).to(x.device)
    Kyy = torch.sum(Kyy) / (n*(n-1))
    
    return Kxx - 2*Kxy + Kyy


def outlier_removal(x, ql, qh):
    ql, qh = np.nanpercentile(x, ql), np.nanpercentile(x, qh)
    print('Percentiles: q_low=%.3f, q_high=%.3f' % (ql, qh))
    x_out = np.copy(x)
    x_out[(x<ql)] = ql
    x_out[(x>qh)] = qh
    return x_out

