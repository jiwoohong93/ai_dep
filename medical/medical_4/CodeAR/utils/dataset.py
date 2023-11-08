import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

GAP_TIME = 6
WINDOW_SIZE = 24
ID_COLS = ['subject_id', 'hadm_id', 'icustay_id']

class Codeset(torch.utils.data.Dataset):
    def __init__(self, data_path, subjects, df_X, df_Y, n_times, n_feats):
        codes = []
        X = []
        Y = []
        for s in subjects:
            code = np.load(f"{data_path}/codes_{s}.npy")
            codes.append(torch.from_numpy(code))
            
            x = df_X[df_X.index.get_level_values('subject_id')==s].to_numpy().reshape(-1, 24, df_X.shape[-1])
            X.append(x)
            
            y = df_Y[df_Y.index.get_level_values('subject_id')==s].to_numpy().astype(int)
            Y.append(y)
            
        self.codes = torch.stack(codes, dim=0)
        
        self.X = np.concatenate(X, axis=0)
        self.M, self.X = torch.Tensor(self.X[:,:,::2].astype(bool)), torch.Tensor(self.X[:,:,1::2].astype(float))
        
        self.Y = torch.LongTensor(np.concatenate(Y, axis=0))
        
    def __getitem__(self, index):
        return (self.codes[index], self.X[index], self.M[index], self.Y[index])

    def __len__(self):
        return len(self.codes)


class MIMICset(torch.utils.data.Dataset):
    def __init__(self, df_X, df_Y, n_times, n_feats):
        self.X = torch.Tensor(df_X.to_numpy().astype(float).reshape(-1, n_times, n_feats))
        self.M, self.X = self.X[:,:,::2], self.X[:,:,1::2]
        self.Y = torch.LongTensor(df_Y.to_numpy().astype(int))
        
    def __getitem__(self, index):
        return (self.X[index], self.M[index], self.Y[index])

    def __len__(self):
        return len(self.X)


def aggregate_data(X, Y):
    X = X[(X.index.get_level_values('icustay_id').isin(set(Y.index.get_level_values('icustay_id')))) &
          (X.index.get_level_values('hours_in') < WINDOW_SIZE)]

    X_subj_idx, Y_subj_idx = [df.index.get_level_values('subject_id') for df in (X, Y)]
    X_subjects = set(X_subj_idx)
    assert X_subjects == set(Y_subj_idx), "Subject ID pools differ!"
    
    X = X[X>0]
    
    idx = pd.IndexSlice
    mask = X.loc[:, idx[:, 'mean']]>0
    mask.rename(columns={'mean': 'mask'}, inplace=True)
    
    X = pd.concat((X, mask), axis=1).loc[:, idx[:, ['mean', 'mask']]]
    X.sort_index(axis=1, inplace=True)
    
    return X, Y

def preprocess_data(df, df_means, df_stds):
    idx = pd.IndexSlice
    df_out = df.copy()
    df_out.loc[:, idx[:,'mean']] = (df_out.loc[:, idx[:,'mean']] - df_means)/df_stds
    df_out.loc[:,idx[:,'mean']] = df_out.loc[:,idx[:,'mean']].groupby(ID_COLS).apply(lambda group: group.interpolate(method='linear'))
    df_out = df_out.fillna(0)
    return df_out
