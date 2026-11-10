import torch
import torch.nn as nn
import torch.distributions.dirichlet as dirichlet
import torch.nn.functional as F
import contextlib
from tqdm import tqdm

from utils import *


def train_full_batch_di(model, inputs, targets, sensitives, optimizer, scheduler, batch_size, lmda, tau, util_criterion, fair_criterion, device) :
    
    """ train for 1 epoch """

    # to GPU
    inputs, targets, sensitives = inputs.float().to(device), targets.to(device), sensitives.to(device)

    # compute the probabilities
    pn0_, pn1_ = get_pn_di(model, inputs, targets, sensitives, device)

    # feed forwarding
    preds, probs = model(inputs)

    # get criterions, compute losses
    util_loss = util_criterion(preds, targets)
    fair_0_ = fair_criterion(pn0_, tau, gamma = 0.5)
    fair_1_ = fair_criterion(pn1_, tau, gamma = 0.5)
    fair_loss = (fair_0_ - fair_1_).abs()

    # GAIF loss
    loss = util_loss + lmda * fair_loss

    # update step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # save losses
    train_loss = loss.item()
   
    return train_loss



######### test 

def test_(model, testloader, device) :

    """ test for 1 epoch """

    test_loss = .0
    n = len(testloader)

    with torch.no_grad() :

        # gather preds
        all_preds = []
        all_targets = []
        all_sensitives = []

        for inputs, targets, sensitives in testloader :

            # current batch size
            mini_size = inputs.shape[0]

            # to GPU
            inputs, targets = inputs.float().to(device), targets.to(device)
            all_targets.append(targets)

            # feed forwarding
            preds, probs = model(inputs)
            all_preds.append(preds)

            # collect sensitives
            all_sensitives.append(sensitives)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_sensitives = torch.cat(all_sensitives)

    return all_preds, all_targets, all_sensitives


def evaluate_di(preds, targets, sensitives) :

    # compute performances
    preds = preds.cpu()
    targets, sensitives = targets.cpu(), sensitives.cpu()

    acc, bacc = util_perf(preds, targets, sensitives)
    di = di_perf(preds, targets, sensitives)
    
    perfs = (acc, bacc, di)

    return perfs


def util_perf(preds, targets, sensitives) :

    # accuracy
    pred_targets = preds.argmax(dim = 1)
    probs = F.softmax(preds, dim = 1)
    acc = (pred_targets == targets).float().mean() 


    acc_g1_ = (preds[targets == 0].argmax(dim = 1) == targets[targets == 0]).float().mean()
    acc_g2_ = (preds[targets == 1].argmax(dim = 1) == targets[targets == 1]).float().mean()
    bacc = (acc_g1_ + acc_g2_) / 2.0

    return round(acc.item(), 4), round(bacc.item(), 4)



def di_perf(preds, targets, sensitives) :

    pred_targets = preds.argmax(dim = 1)

    di0_, di1_ = pred_targets[sensitives == 0].float().mean(), pred_targets[sensitives == 1].float().mean()
    di = (di0_ - di1_).abs()


    return round(di.item(), 4)

