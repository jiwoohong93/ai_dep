import torch.nn as nn


class fair_Loss(nn.Module):
    
    def __init__(self, alg='sipm'):    
        super(fair_Loss, self).__init__()
        self.alg = alg
    
    def forward(self, proj0, proj1):        
        proj0, proj1 = proj0.flatten(), proj1.flatten()
        mean0, mean1 = proj0.mean(), proj1.mean()
        loss = (mean0 - mean1).abs()
            
        return loss
    

