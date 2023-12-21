import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import VQVAE, Discriminator
import hparams as hp
import numpy as np
import pandas as pd
from utils import *
from torch.utils.data import DataLoader
import random
import warnings
warnings.filterwarnings("ignore")

GAP_TIME = 6
WINDOW_SIZE = 24
ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']
DATA_FILEPATH = "./Dataset/all_hourly_data.h5"
PREPROCESSED_DATA_FILEPATH = "./Dataset/preprocessed"
target_vars = ['diastolic blood pressure', 'glucose', 'heart rate', 'oxygen saturation', 'respiratory rate']

def validate(model, val_loader, writer, iteration, discriminator=None):
    model.eval()
    discriminator.eval()
    with torch.no_grad():
        n_data, val_loss_d, val_loss_g, val_recon_loss, val_quantize_loss = 0, 0, 0, 0, 0
        for i, batch in enumerate(val_loader):
            n_data += len(batch[0])
            targets, masks, target_labels = [ x.cuda(non_blocking=True) for x in batch ]
            preds, codes, quantize_loss = model(targets)
            recon_loss = F.mse_loss(torch.masked_select(preds, masks.bool()),
                                    torch.masked_select(targets, masks.bool()))
            
            ####### Discriminator #######
            y, y_hat = discriminator(targets, preds.detach())
            loss_d = ((1-y)**2 + y_hat**2).mean()
            loss_g = ((1-y_hat)**2).mean()

            val_loss_d += loss_d.item() * len(batch[0])
            val_loss_g += loss_g.item() * len(batch[0])
            val_recon_loss += recon_loss.item() * len(batch[0])
            val_quantize_loss += quantize_loss.item() * len(batch[0])
            
        val_loss_d /= n_data
        val_loss_g /= n_data
        val_recon_loss /= n_data
        val_quantize_loss /= n_data

    writer.add_scalar('losses_val/loss_d', val_loss_d, global_step=iteration)
    writer.add_scalar('losses_val/loss_g', val_loss_g, global_step=iteration)
    writer.add_scalar('losses_val/recon_loss', val_recon_loss, global_step=iteration)
    writer.add_scalar('losses_val/quantize_loss', val_quantize_loss, global_step=iteration)
    
    idx = random.randrange(len(targets))
    preds = preds.detach().cpu().numpy()*df_X_stds + df_X_means
    targets = targets.detach().cpu().numpy()*df_X_stds + df_X_means
    for k, v in target_cols.items():
        fig = plot_image(preds[idx, :, v], targets[idx, :, v])
        writer.add_figure(f'plots_{k}/val', fig, global_step=iteration)
    
    model.train()
    discriminator.train()


def main(hp, args):
    global df_X_means
    global df_X_stds
    idx = pd.IndexSlice
    writer = get_writer(hp.output_directory, args.logdir)
        
    X = pd.read_hdf(DATA_FILEPATH, 'vitals_labs')
    statics = pd.read_hdf(DATA_FILEPATH, 'patients')
    Y = statics[statics.max_hours > WINDOW_SIZE + GAP_TIME][['mort_hosp', 'mort_icu', 'los_icu']]
    Y['los_3'] = Y['los_icu'] > 3
    Y['los_7'] = Y['los_icu'] > 7
    Y.drop(columns=['los_icu'], inplace=True)
    Y.astype(float)

    df_X, df_Y = aggregate_data(X, Y)

    train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
    X_subj_idx, Y_subj_idx = [df.index.get_level_values('subject_id') for df in (df_X, df_Y)]
    X_subjects = set(X_subj_idx)
    assert X_subjects == set(Y_subj_idx), "Subject ID pools differ!"

    np.random.seed(0)
    subjects, N = np.random.permutation(list(X_subjects)), len(X_subjects)
    N_train, N_dev, N_test = int(train_frac * N), int(dev_frac * N), int(test_frac * N)
    train_subj = subjects[:N_train]
    dev_subj   = subjects[N_train:N_train + N_dev]
    test_subj  = subjects[N_train+N_dev:]

    [(df_X_train, df_X_dev, df_X_test), (df_Y_train, df_Y_dev, df_Y_test)] = [
        [df[df.index.get_level_values('subject_id').isin(s)] for s in (train_subj, dev_subj, test_subj)] \
        for df in (df_X, df_Y)
    ]

    df_X_means = np.nanmean(df_X_train.loc[:, idx[:, ['mean']]].to_numpy(), axis=0)
    df_X_stds = np.nanstd(df_X_train.loc[:, idx[:, ['mean']]].to_numpy(), axis=0)
    np.save(f'{hp.output_directory}/{args.logdir}/df_X_means.npy', df_X_means)
    np.save(f'{hp.output_directory}/{args.logdir}/df_X_stds.npy', df_X_stds)

    df_X_train = preprocess_data(df_X_train, df_X_means, df_X_stds)
    df_X_dev = preprocess_data(df_X_dev, df_X_means, df_X_stds)
    df_X_test = preprocess_data(df_X_test, df_X_means, df_X_stds)
    print("Data preprocessing is completed!!!")
    
    
    global target_cols
    target_cols = dict([ (col, df_X_train.loc[:, idx[:,'mean']].columns.get_loc((col, 'mean'))) for col in target_vars])
    train_dataset = MIMICset(df_X_train, df_Y_train, n_times=WINDOW_SIZE, n_feats = df_X_train.shape[-1])
    val_dataset = MIMICset(df_X_dev, df_Y_dev, n_times=WINDOW_SIZE, n_feats = df_X_train.shape[-1])
    test_dataset = MIMICset(df_X_test, df_Y_test, n_times=WINDOW_SIZE, n_feats = df_X_train.shape[-1])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=hp.batch_size, drop_last=True)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=hp.batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=hp.batch_size)
    
    model = VQVAE(hp).cuda()
    model.init_codebook(train_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    
    discriminator = Discriminator(hp).cuda()
    discriminator.train()
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=hp.learning_rate)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=0.998)
    
    iteration = 0
    model.train()
    discriminator.train()
    print(f"Training Start!!!")
    for epoch in range(1, hp.epochs+1, 1):
        for i, batch in enumerate(train_loader):
            targets, masks, target_labels = [ x.cuda(non_blocking=True) for x in batch ]
            preds, codes, quantize_loss = model(targets)
            recon_loss = F.mse_loss(torch.masked_select(preds, masks.bool()),
                                    torch.masked_select(targets, masks.bool()))
            
            if iteration==0:
                print(codes.size())
            
            ####### Discriminator #######
            y, y_hat = discriminator(targets, preds.detach())
            loss_d = ((1-y)**2 + y_hat**2).mean()
            optim_d.zero_grad(set_to_none=True)
            loss_d.backward()
            if epoch <= 10:
                optim_d.param_groups[0]['lr'] = hp.learning_rate * (iteration+1) / (len(train_loader)*10)
            optim_d.step()

            ####### Model #######
            _, y_hat = discriminator(targets, preds)
            loss_g = ((1-y_hat)**2).mean()
            optimizer.zero_grad(set_to_none=True)
            (loss_g+recon_loss+quantize_loss).backward()
            if epoch <= 10:
                optimizer.param_groups[0]['lr'] = hp.learning_rate * (iteration+1) / (len(train_loader)*10)
            optimizer.step()
            
            ####### Logging #######
            writer.add_scalar('losses_train/loss_d', loss_d.item(), global_step=iteration)
            writer.add_scalar('losses_train/loss_g', loss_g.item(), global_step=iteration)
            writer.add_scalar('losses_train/recon_loss', recon_loss.item(), global_step=iteration)
            writer.add_scalar('losses_train/quantize_loss', quantize_loss.item(), global_step=iteration)
            iteration += 1

            if iteration%hp.iters_per_checkpoint==0:
                save_checkpoint(model, optimizer, hp.learning_rate, iteration,
                                f'{hp.output_directory}/{args.logdir}/Gen_checkpoint_{iteration}.pt')
                save_checkpoint(discriminator, optim_d, hp.learning_rate, iteration,
                                f'{hp.output_directory}/{args.logdir}/Disc_checkpoint_{iteration}.pt')

        idx = random.randrange(len(targets))
        preds = preds.detach().cpu().numpy()*df_X_stds + df_X_means
        targets = targets.detach().cpu().numpy()*df_X_stds + df_X_means
        for k, v in target_cols.items():
            fig = plot_image(preds[idx, :, v], targets[idx, :, v])
            writer.add_figure(f'plots_{k}/train', fig, global_step=iteration)

        validate(model, val_loader, writer, iteration, discriminator)
        if epoch >= 10:
            scheduler.step()
            scheduler_d.step()
            
    save_checkpoint(model, optimizer, hp.learning_rate, iteration,
                    f'{hp.output_directory}/{args.logdir}/Gen_checkpoint_{iteration}.pt')
    save_checkpoint(discriminator, optim_d, hp.learning_rate, iteration,
                    f'{hp.output_directory}/{args.logdir}/Disc_checkpoint_{iteration}.pt')
            
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('-d', '--logdir', type=str, required=True)
    args = p.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_printoptions(precision=2, sci_mode=False)

    main(hp, args)
