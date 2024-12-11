import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, LinearRegression
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import sys, os


from functions_CSIPM import * 
# from load import *
from load_sIPM import * # 241028 초은


torch.cuda.set_device(2)
 
seed=0


np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True

sim, frac = 5,1
n_SIPM = 10
lr_SIPM = 1
lr_w = 0.1
total_epoch1 = 1000
total_epoch2 = 1
total_epoch3 = 3


_, _, _, _, _, _, _, Cov, x2 = get_data(sim, 0) # X : A C E F,  y : Total_SAT_ACT
(n,d) = Cov.shape




std = StandardScaler()
X_std = std.fit_transform(Cov)

idx = np.where(x2 == 1)[0]

#SIPM
X_c = torch.tensor(np.delete(X_std,idx,axis=0)).float().cuda()
X_t = torch.tensor(X_std[idx,:]).float().cuda()


#############################################################

w_c = w3_(n-len(idx)).cuda()
w_t = w3_(len(idx)).cuda()

para_sigmoid = [para_sigmoid_(d).cuda() for i in range(10)]

SIPM_optimizer = [optim.SGD(para_sigmoid[i].parameters(), lr=lr_SIPM, momentum=0.9, weight_decay=5e-4) for i in range(n_SIPM)]


w_optimizer_c = optim.Adam(w_c.parameters(), lr=lr_w)
w_optimizer_t = optim.Adam(w_t.parameters(), lr=lr_w)

for epoch1 in range(total_epoch1):

    for epoch3 in range(total_epoch3):

        for i in range(n_SIPM):
            loss = -loss2_(X_c, X_t, para_sigmoid[i], w_c, w_t)
            SIPM_optimizer[i].zero_grad()
            loss.backward()
            SIPM_optimizer[i].step()

    for epoch2 in range(total_epoch2):
        loss_list=[]
        for i in range(n_SIPM):
            loss_list.append(loss2_(X_c, X_t, para_sigmoid[i], w_c, w_t))
        loss = max(loss_list)
        
        w_optimizer_c.zero_grad()
        w_optimizer_t.zero_grad()
        loss.backward()
        w_optimizer_c.step()
        w_optimizer_t.step()

w = np.ones(n)
w[idx] = w_t.weight(X_t).detach().cpu().numpy()
w[np.setdiff1d(np.arange(n), idx)] = w_c.weight(X_c).detach().cpu().numpy()
np.save('w_SIPM', w) 



