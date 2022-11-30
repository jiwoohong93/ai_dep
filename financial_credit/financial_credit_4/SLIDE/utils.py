import numpy as np
import torch
import torch.nn as nn



def get_pn_di(model, x, targets, sensitives, device) :

    x = x.float()
    _, probs = model(x)
    
    pn0_, pn1_ = probs[:, 0][sensitives == 0], probs[:, 0][sensitives == 1]

    return pn0_, pn1_
