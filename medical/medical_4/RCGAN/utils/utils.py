import torch
import torch.nn as nn
import torch.nn.functional as F


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


def RBFkernel(X, Y, sigma=1.0):
    return torch.exp(-torch.sum((X-Y)**2)/(2*sigma**2))


def MMD(preds, targets):
    # preds, targets: (B, T, D)
    n=preds.size(0)
    
    kernel_sum1 = 0
    for i in range(n):
        for j in range(n):
            if i!=j:
                kernel_sum1 += RBFkernel(preds[i], preds[j])
    kernel_sum1 /= n*(n-1)
    
    kernel_sum2 = 0
    for i in range(n):
        for j in range(n):
                kernel_sum2 += RBFkernel(preds[i], targets[j])
    kernel_sum2 /= n*n
    
    kernel_sum3 = 0
    for i in range(n):
        for j in range(n):
            if i!=j:
                kernel_sum3 += RBFkernel(targets[i], targets[j])
    kernel_sum3 /= n*(n-1)
    
    return kernel_sum1 - 2*kernel_sum2 + kernel_sum3
    
    
    
    
    