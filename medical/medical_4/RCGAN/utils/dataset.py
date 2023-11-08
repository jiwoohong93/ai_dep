import torch
import hparams as hp
import numpy as np


class eICUDataset(torch.utils.data.Dataset):
    def __init__(self, pid_list, mu, std, seq_list=None):
        self.mu = torch.FloatTensor(mu)
        self.std = torch.FloatTensor(std)
        
        if seq_list is None:
            seq_list = []
            for pid in pid_list:
                seq = np.load(f"./Dataset/physionet.org/files/eicu-crd/2.0/preprocessed/sequences/{pid}")
                seq_list.append(seq)
        self.seq_list = seq_list
        
        self.label_list=[]
        for pid in pid_list:
            label = np.load(f"./Dataset/physionet.org/files/eicu-crd/2.0/preprocessed/labels/{pid}")
            self.label_list.append(label)
        
    def __len__(self):
        return len(self.seq_list)
    
    def __getitem__(self, idx):
        return ( (torch.FloatTensor(self.seq_list[idx])-self.mu)/self.std, torch.LongTensor(self.label_list[idx]) )