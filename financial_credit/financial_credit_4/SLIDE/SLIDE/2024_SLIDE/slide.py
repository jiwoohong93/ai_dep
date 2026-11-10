import torch
import torch.nn as nn

""" Fairness Constraints """

class fair_penalty(nn.Module) :

    def __init__(self, mode = "slide", gamma = 0.5, tau = 0.1) :
        super(fair_penalty, self).__init__()
        
        self.mode = mode
        self.gamma = gamma
        self.tau = tau
        print("gamma: {}, tau : {}".format(self.gamma, self.tau))
        self.ReLU = nn.ReLU()

    def forward(self, pn, tau, gamma) :
        assert gamma == self.gamma

        if self.mode == "slide" :
            term1_ = self.ReLU(pn - gamma) / tau
            term2_ = self.ReLU(pn - gamma - tau) / tau
            loss = term1_ - term2_

        elif self.mode == "hinge" :
            hinge = self.ReLU(pn - gamma + 1)
            loss = hinge

        else :
            print("No other surrogate losses considered")
            raise NotImplementedError

        return loss.mean()

def Indicator(x) :
    x[x <= 0.0] = 0.0
    x[x > 0.0] = 1.0
    return x

def true_nu(z, gamma, reduction = "mean") :

    if reduction == "none" :
        return Indicator(z - gamma)
    elif reduction == "mean" :
        return Indicator(z - gamma).mean()
    elif reduction == "sum" :
        return Indicator(z - gamma).sum()
    else :
        print("true fairness loss error")
        raise NotImplementedError
