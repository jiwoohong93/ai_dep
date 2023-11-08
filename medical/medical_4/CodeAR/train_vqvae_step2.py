import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import CodeARmodel, VQVAE
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

def validate(model, vqvae, val_loader, writer, iteration):
    model.eval()
    with torch.no_grad():
        n_data, val_loss_d, val_loss_g, val_recon_loss, val_quantize_loss = 0, 0, 0, 0, 0
        for i, batch in enumerate(val_loader):
            n_data += len(batch[0])
            targets, real_targets, masks, target_labels = [ x.cuda(non_blocking=True) for x in batch ]
            preds = model(targets, target_labels)
            loss_g = F.nll_loss(preds.view(-1, preds.size(-1)), targets.view(-1))
            
            val_loss_g += loss_g.item() * len(batch[0])
            
        val_loss_g /= n_data

    writer.add_scalar('losses_val/loss_g', val_loss_g, global_step=iteration)
    
    with torch.no_grad():
        generated = model.inference(target_labels)
        idx = random.randrange(len(targets))
        
        preds = vqvae.decode(torch.argmax(preds, dim=-1))
        targets = vqvae.decode(targets)
        generated = vqvae.decode(generated)

    val_loss_mmd = MMD(generated, real_targets).item()
    writer.add_scalar('losses_val/loss_mmd', val_loss_mmd, global_step=iteration)

    preds = preds.detach().cpu().numpy()*df_X_stds + df_X_means
    targets = targets.detach().cpu().numpy()*df_X_stds + df_X_means
    real_targets = real_targets.detach().cpu().numpy()*df_X_stds + df_X_means
    generated = generated.detach().cpu().numpy()*df_X_stds + df_X_means
    for k, v in target_cols.items():
        fig = plot_image(preds[idx, :, v], targets[idx, :, v], generated[idx, :, v])
        writer.add_figure(f'plots_{k}/val', fig, global_step=iteration)

        fig = plot_image(targets[idx, :, v], real_targets[idx, :, v])
        writer.add_figure(f'plots_{k}/recon_val', fig, global_step=iteration)

    model.train()


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

    df_X_train = preprocess_data(df_X_train, df_X_means, df_X_stds)
    df_X_dev = preprocess_data(df_X_dev, df_X_means, df_X_stds)
    df_X_test = preprocess_data(df_X_test, df_X_means, df_X_stds)
    print("Data preprocessing is completed!!!")


    global target_cols
    target_cols = dict([ (col, df_X_train.loc[:, idx[:,'mean']].columns.get_loc((col, 'mean'))) for col in target_vars])
    train_dataset = Codeset(f"./Dataset/codes/{args.vqvaedir}",
                            train_subj, df_X_train, df_Y_train, n_times=WINDOW_SIZE, n_feats = df_X_train.shape[-1])
    val_dataset = Codeset(f"./Dataset/codes/{args.vqvaedir}",
                          dev_subj, df_X_dev, df_Y_dev, n_times=WINDOW_SIZE, n_feats = df_X_train.shape[-1])
    test_dataset = Codeset(f"./Dataset/codes/{args.vqvaedir}",
                           test_subj, df_X_test, df_Y_test, n_times=WINDOW_SIZE, n_feats = df_X_train.shape[-1])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=hp.batch_size, drop_last=True)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=hp.batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=hp.batch_size)
    
    model = CodeARmodel(hp).cuda()
    vqvae = VQVAE(hp).cuda()
    checkpoint_dict = torch.load(f"./training_log/{args.vqvaedir}/Gen_checkpoint_149000.pt", map_location='cpu')
    vqvae.load_state_dict(checkpoint_dict['state_dict'])
    vqvae.eval()
    
    init_embed = vqvae.Quantizer.q_linear(F.normalize(vqvae.Quantizer.codebook, p=2, dim=-1))
    model.embedding.weight.data.copy_(init_embed)
    model.proj.weight.data.copy_(init_embed)
    model.proj.bias.data.zero_()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    torch.save(vqvae.state_dict(), f'{hp.output_directory}/{args.logdir}/vqvae.pt')

    iteration = 0
    model.train()
    print(f"Training Start!!!")
    for epoch in range(1, hp.epochs+1, 1):
        for i, batch in enumerate(train_loader):
            targets, real_targets, masks, target_labels = [ x.cuda(non_blocking=True) for x in batch ]
            preds = model(targets, target_labels)
            loss_g = F.nll_loss(preds.view(-1, preds.size(-1)), targets.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss_g.backward()
            if epoch <= 10:
                optimizer.param_groups[0]['lr'] = hp.learning_rate * (iteration+1) / (len(train_loader)*10)
            optimizer.step()
            
            ####### Logging #######
            writer.add_scalar('losses_train/loss_g', loss_g.item(), global_step=iteration)
            iteration += 1

            if iteration%hp.iters_per_checkpoint==0:
                save_checkpoint(model, optimizer, hp.learning_rate, iteration,
                                f'{hp.output_directory}/{args.logdir}/Gen_checkpoint_{iteration}.pt')

            idx = random.randrange(len(targets))
            with torch.no_grad():
                preds = vqvae.decode(torch.argmax(preds, dim=-1))
                targets = vqvae.decode(targets)
            
        preds = preds.detach().cpu().numpy()*df_X_stds + df_X_means
        targets = targets.detach().cpu().numpy()*df_X_stds + df_X_means
        real_targets = real_targets.detach().cpu().numpy()*df_X_stds + df_X_means
        for k, v in target_cols.items():
            fig = plot_image(preds[idx, :, v], targets[idx, :, v])
            writer.add_figure(f'plots_{k}/train', fig, global_step=iteration)
            
            fig = plot_image(targets[idx, :, v], real_targets[idx, :, v])
            writer.add_figure(f'plots_{k}/recon_train', fig, global_step=iteration)

        validate(model, vqvae, val_loader, writer, iteration)
        if epoch >= 10:
            scheduler.step()
            
    save_checkpoint(model, optimizer, hp.learning_rate, iteration,
                    f'{hp.output_directory}/{args.logdir}/Gen_checkpoint_{iteration}.pt')
    
    ### Test MMD ###
    model.eval()
    vqvae.eval()
    with torch.no_grad():
        target_list, pred_list = [], []
        for i, batch in enumerate(test_loader):
            _, real_targets, _, target_labels = [ x.cuda(non_blocking=True) for x in batch ]
            target_list.append(real_targets)
            generated = model.inference(target_labels)
            generated = vqvae.decode(generated)
            pred_list.append(generated)
            
        targets = torch.cat(target_list, dim=0)
        generated = torch.cat(pred_list, dim=0)
        test_loss_mmd = MMD(generated, targets).item()
        writer.add_scalar('losses_test/loss_mmd', test_loss_mmd, global_step=iteration)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--vqvaedir', type=str, required=True)
    p.add_argument('-d', '--logdir', type=str, required=True)
    args = p.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_printoptions(precision=2, sci_mode=False)

    main(hp, args)
