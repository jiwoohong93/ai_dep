import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import VQVAE
import hparams as hp
from utils import *
from torch.utils.data import DataLoader
import IPython.display as ipd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

GAP_TIME = 6
WINDOW_SIZE = 24
ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']
DATA_FILEPATH = "./Dataset/all_hourly_data.h5"
    
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--vqvaedir', type=str, default='')
    args = p.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    model = VQVAE(hp).cuda()
    checkpoint_dict = torch.load(f"./training_log/{args.vqvaedir}/Gen_checkpoint_149000.pt", map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    model.eval()

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

    idx = pd.IndexSlice
    df_X_means = np.nanmean(df_X_train.loc[:, idx[:, ['mean']]].to_numpy(), axis=0)
    df_X_stds = np.nanstd(df_X_train.loc[:, idx[:, ['mean']]].to_numpy(), axis=0)

    df_X = preprocess_data(df_X, df_X_means, df_X_stds)
    df_X = df_X.loc[:, idx[:,'mean']]

    if not os.path.exists(f"./Dataset/codes/{args.vqvaedir}"):
        os.makedirs(f"./Dataset/codes/{args.vqvaedir}", exist_ok=True)

    for i, subj in enumerate(tqdm(X_subjects)):
        x = torch.Tensor(df_X[df_X.index.get_level_values('subject_id')==subj].to_numpy().reshape(-1, 24, df_X.shape[-1]))
        with torch.no_grad():
            codes = model.quantize(x.cuda())
            if i==0:
                print(codes.shape)
                print(codes)

        np.save(f"./Dataset/codes/{args.vqvaedir}/codes_{subj}.npy", codes[0].detach().cpu().numpy())
            