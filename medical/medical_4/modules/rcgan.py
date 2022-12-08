import torch
import torch.nn as nn
import torch.nn.functional as F


class RCGAN(nn.Module):
    def __init__(self, hp):
        super(RCGAN, self).__init__()
        self.hp = hp
        self.label_linear = nn.Sequential(nn.Linear(hp.label_dim, hp.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hp.hidden_dim, hp.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hp.hidden_dim, hp.hidden_dim, bias=False))
        self.lstm = nn.LSTMCell(hp.hidden_dim, hp.hidden_dim)
        self.proj = nn.Sequential(nn.Linear(hp.hidden_dim, hp.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hp.hidden_dim, hp.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hp.hidden_dim, hp.orig_dim))
        
    def forward(self, labels):
        conds = self.label_linear(labels.float())
        
        x_pred = []
        h_t = torch.zeros_like(conds)
        c_t = torch.zeros_like(h_t)
        for i in range(self.hp.fixed_len):
            z = torch.randn_like(h_t) # B, D
            h_t, c_t = self.lstm(conds+z, (h_t, c_t))
            x_t = self.proj(h_t)
            x_pred.append(x_t)
            
        return torch.stack(x_pred, dim=1)
