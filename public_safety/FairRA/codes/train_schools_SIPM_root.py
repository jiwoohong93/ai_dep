import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import math
import random
import time
import load_sIPM

os.environ["MKL_NUM_THREADS"] = "3" 
os.environ["NUMEXPR_NUM_THREADS"] = "3"  
os.environ["OMP_NUM_THREADS"] = "3" 

# from this: http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class the_linear(nn.Module):
    def __init__(self, d):
        super(the_linear, self).__init__()
        self.lin = nn.Linear(d, 1, bias=False)
    def forward(self, X):
        P = self.lin(X)
        return P

# X - (n,d), this is the concatenation of the matrices
def train(model, n_epochs, X, y, w = None , print_every=1):
    random.seed(0)
    start = time.time()
    optimizer = optim.LBFGS(model.parameters())
    
    if w == None:
        loss_func = nn.MSELoss()
    else:
        loss_func = nn.MSELoss(reduction='none')

    for i in range(1, n_epochs + 1):
        def closure():
            optimizer.zero_grad()
            p = model(X)
            if w == None:
                loss = loss_func(p, y)
            else:            
                loss = torch.sum(loss_func(p, y)*w)
            loss.backward()
            return loss
        optimizer.step(closure)
        
        
        with torch.no_grad():
            p = model(X)
            loss = torch.sum(loss_func(p, y)*w)
            print('train loss:', loss.item())
    # EVAL MODEL HERE!
    return model


def train2(model, n_epochs, X, y, w = None , print_every=1):
    random.seed(0)
    start = time.time()
    optimizer = optim.Adam(model.parameters(), lr=1e-1)
    
    if w == None:
        loss_func = nn.MSELoss()
    else:
        loss_func = nn.MSELoss(reduction='none')

    for i in range(1, n_epochs + 1):
        optimizer.zero_grad()
        p = model(X)
        if w == None:
            loss = loss_func(p, y)
        else:            
            loss = torch.sum(loss_func(p, y).squeeze()*w)
        loss.backward()
        optimizer.step()
        
        print(loss.item())
#        with torch.no_grad():
#            p = model(X)
#            loss = torch.sum(loss_func(p, y)*w)
#            print('train loss:', loss.item())

def all_of_it(sim, frac):
    use_cuda = torch.cuda.is_available()
    CUDA_GPU = 2
    # get data
    print('sim: ' + str(sim))
    print('frac: ' + str(frac))
    X, y, _, _, _, _, _, Cov, x2 = load_sIPM.get_data(sim, 0) # X : A C E F,  y : Total_SAT_ACT
    (n,d) = X.shape
    X_t = Variable(torch.FloatTensor(X))
    y_t = Variable(torch.FloatTensor(y))
    if use_cuda:
        X_t = X_t.cuda(CUDA_GPU)
        y_t = y_t.cuda(CUDA_GPU)
    # model
    model = nn.Sequential(nn.Linear(d,1,bias=False))
    model = the_linear(d)
    if use_cuda:
        model = model.cuda(CUDA_GPU)
    w_new = torch.tensor(np.load('w_SIPM.npy')).cuda(CUDA_GPU)
    w_new = torch.sqrt(w_new)
    w_new = w_new/sum(w_new)
    n_epochs = 100
    train2(model, n_epochs, X_t, y_t, w_new)
#    train2(model, n_epochs, X_t, y_t, None)
    weights = list(model.parameters())[0]
    weights1 = weights.detach().cpu().numpy()
    if frac:
        np.savez_compressed('school_weights_linear_mse_sim' + str(sim) + '_max1_frac_sIPM_root_res', w=weights1)
    else:
        np.savez_compressed('school_weights_linear_mse_sim' + str(sim) + '_max1_sIPM_root_res', w=weights1)

if __name__ == '__main__':
    all_of_it(5, 1)
    
    

