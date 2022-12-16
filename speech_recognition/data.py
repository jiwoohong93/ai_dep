import pickle
import numpy as np
import torch
import json
import pandas as pd
import os
from os.path import join
from math import ceil
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from utils import extract_feature
from easydict import EasyDict as edict
from tqdm import tqdm
from collections import Counter
from pprint import pprint

class CORAALTranscriptionDatasetSattrIndex(Dataset):
    def __init__(self, mode, config):
        
        self.config = config
        self.config.verbose = self.config.get('verbose', True)
        self.mode = mode
        self.duration = self.config.data.min_duration
        self.sattr = self.config.data.get('sensitive_attr', 'dialect_int')

        df = pd.read_csv(self.config[mode].meta_data)
        df = df[df.end-df.start>=self.duration]

        # extract gender info from file name
        df['gender'] = [f.split('/')[-1].split('_')[-3] for f in df['file']]
        df['gender_int'] = [0 if g=='m' else 1 for g in df['gender']]
        if self.config.verbose:
            print('sample size ', self.config.data.sample_size)
        seed = self.config.train.get('seed', 2020)
        if self.config.verbose:
            print(f'{mode} dataset seed {seed}')
        if self.mode == 'train':
            sample_size = min(self.config.data.sample_size, len(df))

            self.df = df.sample(sample_size, random_state=seed)
        else:
            sample_size = len(df)
            sample_size = min(self.config.data.sample_size, len(df))
            self.df = df.sample(sample_size, random_state=seed)

        if self.config.verbose:
            print(f'DialectTranscriptionDataset size {len(self.df)}')
            print(self.df.head(2))

            from collections import Counter
            cnt = Counter(self.df[self.sattr])
            print(self.sattr, cnt)

        self.df['id'] = list(range(len(self.df)))

    def pad_collate(self, batch):
        max_input_len = float('-inf')
        max_target_len = float('-inf')

        for elem in batch:
            index, sattr, feature, trn = elem
            max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]
            max_target_len = max_target_len if max_target_len > len(trn) else len(trn)

        for i, elem in enumerate(batch):
            index, sattr, f, trn = elem
            input_length = f.shape[0]
            input_dim = f.shape[1]
            feature = np.zeros((max_input_len, input_dim), dtype=np.float)
            feature[:f.shape[0], :f.shape[1]] = f
            trn = np.pad(trn, (0, max_target_len - len(trn)), 'constant', constant_values=self.config.data.PAD_token)
            batch[i] = (index, sattr, feature, trn, input_length)

        batch.sort(key=lambda x: x[4], reverse=True)

        return default_collate(batch)
        
    def extract_feature_path(self, sample):
        wave = sample['file']
        start_time, duration_time = sample.start, sample.duration
        save_name = f"{wave.replace('/','-')}-{'{:.2f}'.format(start_time)}-{'{:.2f}'.format(duration_time)}"
        save_dir = join(self.config.feature.stft_dir, save_name)
        return save_dir

    def __getitem__(self, i):

        sample = self.df.iloc[i]

        # speaker_id = self.speaker2id[sample.speaker]
        trn = list(map(lambda x: int(x), sample['trans_int'].strip().split()))
        feature_path = self.extract_feature_path(sample)

        feature = np.load(f'{feature_path}.npy')
        sattr = sample[self.sattr]
        index = sample['id']
        return index, sattr, feature, trn, 

    def __len__(self):
        return len(self.df)


class DialectPlusDatasetDialect(Dataset):
    def __init__(self, mode, config):
        self.config = config
        self.mode = mode
        self.duration = self.config.data.min_duration
        df = pd.read_csv(self.config[mode].meta_data)
        df = df[df.end-df.start>=self.duration]
        seed = self.config.train.seed if 'seed' in self.config.train else 2020
        sample_size = min(self.config.data.sample_size, len(df))
        self.df = df.sample(sample_size, random_state=seed)

        print(f'{mode} dataset seed: {seed} size: {len(self.df)} max: {max(df.duration):.4f} mean: {np.mean(df.duration):.4f} std: {np.std(df.duration):.4f}')

        from collections import Counter
        cnt = Counter(self.df.dialect)
        pprint(dict(cnt))

    def pad_collate(self, batch):
        max_input_len = float('-inf')
        max_target_len = float('-inf')

        for elem in batch:
            index, dialect, feature, trn = elem
            max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]
            max_target_len = max_target_len if max_target_len > len(trn) else len(trn)

        for i, elem in enumerate(batch):
            index, dialect, f, trn = elem
            input_length = f.shape[0]
            input_dim = f.shape[1]
            feature = np.zeros((max_input_len, input_dim), dtype=np.float)
            feature[:f.shape[0], :f.shape[1]] = f
            trn = np.pad(trn, (0, max_target_len - len(trn)), 'constant', constant_values=self.config.data.PAD_token)
            batch[i] = (index, dialect, feature, trn, input_length)

        batch.sort(key=lambda x: x[4], reverse=True)

        return default_collate(batch)

    def extract_feature_path(self, sample):
        # wave = sample['file']
        # start_time, duration_time = sample.start, sample.duration
        # save_name = f"{self.mode}-{sample.id}"
        save_name = f"{self.mode}-{sample.id}-{len(sample.trans)}"
        save_dir = join(self.config.feature.stft_dir, save_name)
        return save_dir

    def __getitem__(self, i):
        sample = self.df.iloc[i]
        trn = list(map(lambda x: int(x), sample['trans_int'].strip().split()))
        feature_path = self.extract_feature_path(sample)

        feature = np.load(f'{feature_path}.npy')
        dialect = sample['dialect_int']
        index = sample['id']
        return index, dialect, feature, trn, 

    def __len__(self):
        return len(self.df)
