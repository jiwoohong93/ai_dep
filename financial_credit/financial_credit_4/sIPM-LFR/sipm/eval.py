import numpy as np
import torch
from sklearn.metrics import (accuracy_score, 
                             balanced_accuracy_score,
                             average_precision_score
                            )

from sipm.train import _loss


def _eval(dataset=None,
          model=None, aud_model=None,
          lmda=None, lmdaF=None, lmdaR=None,
          criterion=None, fair_criterion=None):
    
    # to eval
    model.eval()
    x, y, s = dataset
    x, y, s = x.cuda(), y.cuda(), s.cuda()
    with torch.no_grad():
        z = model.encoder(x)
        logits, preds = model.head(z)
        recon = model.decoder(z)
        logits = logits.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        
        loss = 0.0
        if criterion is not None:
            loss = _loss(x, y, s,
                         lmda, lmdaF, lmdaR,
                         model, aud_model,
                         criterion, fair_criterion)
            loss = loss.item()
        
    s = s.flatten().cpu().numpy().astype(int)
    y = y.flatten().cpu().numpy().astype(int)
        
    """ utility """
    preds = preds.flatten()
    pred_labels = (preds > 0.5).astype(int)    
    # acc
    acc = accuracy_score(y, pred_labels)    
    # bacc
    bacc = balanced_accuracy_score(y, pred_labels)    
    # ap
    ap = average_precision_score(y, preds)
    
    """ fairness """
    preds0, preds1 = preds[s == 0], preds[s == 1]      
    taus = np.arange(0.0, 1.0, 0.01)    
    # dp
    dp = (preds0 > 0.5).mean() - (preds1 > 0.5).mean()
    dp = abs(dp)
    # mdp
    mdp = preds0.mean() - preds1.mean()
    mdp = abs(mdp) 
    # vdp
    vdp = preds0.std()**2 - preds1.std()**2
    vdp = abs(vdp)    
    # sdp
    dps = []
    for tau in taus:
        tau_dp = (preds0 > tau).mean() - (preds1 > tau).mean()
        dps.append(abs(tau_dp))
    sdp = np.mean(dps)
        
    return {"loss": loss, 
            "acc": acc, "bacc": bacc, "ap": ap,
            "dp": dp, "mdp": mdp, "vdp": vdp, "sdp": sdp
            }
