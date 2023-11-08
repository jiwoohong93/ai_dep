import torch
import torch.nn as nn
import torch.nn.functional as F


class CodeARmodel(nn.Module):
    def __init__(self, hp):
        super(CodeARmodel, self).__init__()
        self.hp = hp
        self.hp.fixed_len = self.hp.fixed_len//hp.stride
            
        self.embedding = nn.Embedding(hp.n_codes, hp.hidden_dim)
        self.x_linear = nn.Sequential(nn.Linear(hp.hidden_dim, hp.hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hp.hidden_dim, hp.hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hp.hidden_dim, hp.hidden_dim, bias=False))
        self.label_linear = nn.Sequential(nn.Linear(hp.label_dim, hp.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hp.hidden_dim, hp.hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hp.hidden_dim, hp.hidden_dim, bias=False))
        self.lstm1 = nn.LSTMCell(hp.hidden_dim, hp.hidden_dim)
        self.lstm2 = nn.LSTMCell(hp.hidden_dim, hp.hidden_dim)
        self.proj = nn.Linear(hp.hidden_dim, hp.n_codes)
        
        self.sos = nn.Parameter(torch.randn(1, 1, hp.hidden_dim))
            

    def forward(self, x, labels):
        B, T = x.size()
        x = self.x_linear(self.embedding(x))
        x_shift = torch.cat([self.sos.repeat(B, 1, 1), x[:, :-1]], dim=1)
        conds = self.label_linear(labels.float())
        
        x_pred = []
        h1_t = torch.zeros_like(conds)
        h2_t = torch.zeros_like(conds)
        c1_t = torch.zeros_like(conds)
        c2_t = torch.zeros_like(conds)
        for i in range(self.hp.fixed_len):
            h1_t, c1_t = self.lstm1(F.dropout(conds+x_shift[:, i], 0.5, True), (h1_t, c1_t))
            h2_t, c2_t = self.lstm2(F.dropout(h1_t, 0.5, True), (h2_t, c2_t))
            x_t = self.proj(h2_t)
            x_t = F.log_softmax(x_t, dim=-1)
            x_pred.append(x_t)
            
        return torch.stack(x_pred, dim=1)
    
    
    def inference(self, labels):
        conds = self.label_linear(labels.float())
        
        x_pred = []
        h1_t = torch.zeros_like(conds)
        h2_t = torch.zeros_like(conds)
        c1_t = torch.zeros_like(conds)
        c2_t = torch.zeros_like(conds)
        for i in range(self.hp.fixed_len):
            if i==0:
                x_prev = self.sos.squeeze(1)
            else:
                x_prev = self.x_linear( self.embedding(x_pred[-1]) )
                
            h1_t, c1_t = self.lstm1(F.dropout(conds+x_prev, 0.5, True), (h1_t, c1_t))
            h2_t, c2_t = self.lstm2(F.dropout(h1_t, 0.5, True), (h2_t, c2_t))
            x_t = self.proj(h2_t)
            
            x_pred.append(self.top_p_nucleus_sampling_batch(x_t))
        return torch.stack(x_pred, dim=1)

    def top_p_nucleus_sampling_batch(self, logits, p=0.9):
        batch_size, vocab_size = logits.size()
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        index = torch.where(cumulative_probs >= p)[1]
        filtered_logits = torch.zeros(batch_size, vocab_size).to(logits.device)
        for i in range(batch_size):
            j = index[i]
            filtered_logits[i, :j + 1] = sorted_logits[i, :j + 1]
            filtered_logits[i, j + 1:] = -float('inf')

        sample_index = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)[:,0]

        return torch.gather(sorted_indices, dim=-1, index=sample_index.unsqueeze(-1)).squeeze(-1)
