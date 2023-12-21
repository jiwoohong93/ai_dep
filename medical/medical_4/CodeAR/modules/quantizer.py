import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import *
from utils import *


class Quantizer(nn.Module):
    def __init__(self, hp):
        super(Quantizer, self).__init__()
        self.n_codes = hp.n_codes
        self.hidden_dim = hp.hidden_dim
        
        self.init=False
        self.codebook = nn.Parameter(torch.randn(self.n_codes, hp.bottleneck_dim))
        
        self.z_linear = nn.Linear(hp.hidden_dim, hp.bottleneck_dim, bias=False)
        self.q_linear = nn.Linear(hp.bottleneck_dim, hp.hidden_dim, bias=False)

    def forward(self, z):
        B, T, D = z.size()
        if self.training==False:
            self.init=True

        z_norm = F.normalize(self.z_linear(z), p=2, dim=-1) # B, T, d
        code_norm = F.normalize(self.codebook, p=2, dim=-1) # n_codes, d
        with torch.no_grad():
            z_flatten = z_norm.reshape(-1, z_norm.size(-1))
            dist = (z_flatten.pow(2).sum(1, keepdim=True)
                    - 2*z_flatten@code_norm.t()
                    + code_norm.t().pow(2).sum(0, keepdim=True))
            codes = torch.argmin(dist, dim=-1).reshape(B, T)
            y = F.one_hot(codes, self.n_codes).float().reshape(-1, self.n_codes) # B*T, n_codes
            
        q_norm = (y @ code_norm).reshape(B, T, -1)

        quantize_loss = F.mse_loss(q_norm, z_norm.detach()) + 0.25*F.mse_loss(q_norm.detach(), z_norm)
        q = self.q_linear(z_norm + (q_norm-z_norm).detach())

        return q, codes, quantize_loss
