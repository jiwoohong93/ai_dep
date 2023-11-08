import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import RCGAN, Discriminator
import hparams as hp
import numpy as np
import pandas as pd
from utils import *
from torch.utils.data import DataLoader
import random


def validate(model, discriminator, val_loader, writer, iteration):
    model.eval()
    discriminator.eval()
    with torch.no_grad():
        n_data, val_loss_d, val_loss_g = 0, 0, 0
        for i, batch in enumerate(val_loader):
            targets, target_labels = [ x.cuda(non_blocking=True) for x in batch ]
            n_data += len(targets)
            preds = model(target_labels)
            
            y, y_hat = discriminator(targets, preds.detach(), target_labels)
            loss_d = ((1-y)**2 + y_hat**2).mean()
            
            _, y_hat = discriminator(targets, preds, target_labels)
            loss_g = ((1-y_hat)**2).mean()

            val_loss_d += loss_d.item() * len(targets)
            val_loss_g += loss_g.item() * len(targets)

        val_loss_d /= n_data
        val_loss_g /= n_data
    
    model.train()
    discriminator.train()
    
    val_loss_mmd = MMD(preds, targets).item()
    writer.add_scalar('losses_val/loss_mmd', val_loss_mmd, global_step=iteration)
    writer.add_scalar('losses_val/loss_d', val_loss_d, global_step=iteration)
    writer.add_scalar('losses_val/loss_g', val_loss_g, global_step=iteration)
    
    idx = random.randrange(len(targets))
    for i in range(hp.orig_dim):
        fig = plot_image(preds[idx, :, i].detach().cpu(), targets[idx, :, i].detach().cpu())
        writer.add_figure(f'plots_{i}/val', fig, global_step=iteration)


def main(hp, args):
    pid_list = os.listdir("./Dataset/physionet.org/files/eicu-crd/2.0/preprocessed/sequences")
    random.seed(1234)
    random.shuffle(pid_list)
    train_pid = pid_list[:int(0.8*len(pid_list))]
    val_pid = pid_list[int(0.8*len(pid_list)):int(0.9*len(pid_list))]
    test_pid = pid_list[int(0.9*len(pid_list)):]
    
    seq_list = []
    for pid in train_pid:
        seq = np.load(f"./Dataset/physionet.org/files/eicu-crd/2.0/preprocessed/sequences/{pid}")
        seq_list.append(seq)
    mu, std = np.concatenate(seq_list, axis=0).mean(axis=0), np.concatenate(seq_list, axis=0).std(axis=0)
    
    train_dataset = eICUDataset(train_pid, mu, std, seq_list=seq_list)
    val_dataset = eICUDataset(val_pid, mu, std)
    test_dataset = eICUDataset(test_pid, mu, std)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=hp.batch_size, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=hp.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=hp.batch_size)
    
    model = RCGAN(hp).cuda()
    discriminator = Discriminator(hp, conditional=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=hp.learning_rate, betas=[0.5, 0.9])
    writer = get_writer(hp.output_directory, args.logdir)

    iteration = 0
    model.train()
    discriminator.train()
    for epoch in range(1, hp.epochs+1, 1):
        for i, batch in enumerate(train_loader):
            targets, target_labels = [ x.cuda(non_blocking=True) for x in batch ]
            preds = model(target_labels)
            
            ####### Discriminator #######
            y, y_hat = discriminator(targets, preds.detach(), target_labels)
            loss_d = ((1-y)**2 + y_hat**2).mean()
            optim_d.zero_grad(set_to_none=True)
            loss_d.backward()
            optim_d.step()
            
            ####### Model #######
            _, y_hat = discriminator(targets, preds, target_labels)
            loss_g = ((1-y_hat)**2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss_g.backward()
            optimizer.step()
            
            ####### Logging #######
            writer.add_scalar('losses_train/loss_d', loss_d.item(), global_step=iteration)
            writer.add_scalar('losses_train/loss_g', loss_g.item(), global_step=iteration)

            iteration += 1
            
            if iteration%hp.iters_per_checkpoint==0:
                save_checkpoint(model, optimizer, hp.learning_rate, iteration,
                                f'{hp.output_directory}/{args.logdir}/RCGAN_checkpoint_{iteration}.pt')
                save_checkpoint(discriminator, optim_d, hp.learning_rate, iteration,
                                f'{hp.output_directory}/{args.logdir}/Disc_checkpoint_{iteration}.pt')
                
        loss_mmd = MMD(preds, targets)
        writer.add_scalar('losses_train/loss_mmd', loss_mmd.item(), global_step=iteration) 
        
        idx = random.randrange(len(targets))
        for i in range(hp.orig_dim):
            fig = plot_image(preds[idx, :, i].detach().cpu(), targets[idx, :, i].detach().cpu())
            writer.add_figure(f'plots_{i}/train', fig, global_step=iteration)
        
        validate(model, discriminator, val_loader, writer, iteration)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--logdir', type=str, required=True)
    args = p.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_printoptions(precision=2, sci_mode=False)

    main(hp, args)
