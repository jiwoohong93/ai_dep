import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as sn
from torch.nn.utils import weight_norm as wn

class Discriminator(nn.Module):
    def __init__(self, hp):
        super(Discriminator, self).__init__()
        self.proj_up = sn(nn.Linear(hp.orig_dim, hp.hidden_dim))
        self.label_linear = nn.Sequential(nn.Linear(hp.label_dim, hp.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hp.hidden_dim, hp.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hp.hidden_dim, hp.hidden_dim, bias=False))
        self.lstm = nn.LSTM(hp.hidden_dim, hp.hidden_dim, 1, batch_first=True)
        self.proj_down = wn(nn.Linear(hp.hidden_dim, 1))
            
    def forward(self, x, x_hat):
        x = F.dropout(self.proj_up(torch.cat([x, x_hat], dim=0)), 0.5, self.training) # 2*B, T, D
        x, _ = self.lstm(x)
        y, y_hat = (self.proj_down(x).squeeze(-1)).chunk(2, dim=0)
        
        return y, y_hat