import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import sys, os
import torch.nn.functional as F

def expit(x):
    return 1 / (1 + np.exp(-1 * x))

def loss2_(X_1, X_2, para_sigmoid, w_1, w_2): # for ATE
    
    n_1, n_2 = X_1.shape[0], X_2.shape[0]

    sig_1 = para_sigmoid(X_1)
    sig_2 = para_sigmoid(X_2)

    
    weight_1 = w_1.weight(X_1)    
    weight_2 = w_2.weight(X_2)      
    
    out1 = (torch.sum(sig_1)+torch.sum(sig_2))/(n_1+n_2) - torch.sum(sig_1 * weight_1)
    out2 = (torch.sum(sig_1)+torch.sum(sig_2))/(n_1+n_2) - torch.sum(sig_2 * weight_2)
    return out1**2 + out2**2
  
class para_sigmoid_(nn.Module):
    def __init__(self, input_dim):
        super(para_sigmoid_, self).__init__()
                
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim,1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):              
        x = self.linear(x)
        x = self.sigmoid(x.squeeze())
        return x
    
    
class w3_(nn.Module):
    def __init__(self, n):
        super(w3_, self).__init__()
                
        self.vec = nn.Parameter(torch.ones(n))
        self.softmax = nn.Softmax(0)
    
    def weight(self, x):
        x = self.softmax(self.vec)
        return x
    
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__