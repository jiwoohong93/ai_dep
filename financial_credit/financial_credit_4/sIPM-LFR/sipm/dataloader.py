import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from datasets.adult import AdultDataset
from datasets.compas import CompasDataset
from datasets.health import HealthDataset


class _dataloader:
    
    def __init__(self, dataset, batch_size, scaling):
        """_summary_
        Args:
            dataset (str): select one from {adult, compas, health}
            batch_size (int): batch size
            scaling (bool): scaling input vectors for reconstruction
        """
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.scaling = scaling
        self.scaler = MinMaxScaler()
        
        if self.dataset == "adult":
            self.data = AdultDataset()
        elif self.dataset == "compas":
            self.data = CompasDataset()
        elif self.dataset == "health":
            self.data = HealthDataset()
        
    def _to_dataloader(self, data, batch_size):
        
        if self.dataset == 'adult':            
            # get train datasets
            df = data.train
            x_idx = df.columns.values.tolist()
            x_idx.remove("result")
            if self.scaling:
                x = torch.from_numpy(self.scaler.fit_transform(df[x_idx].values)).type(torch.float)
            else:
                x = torch.from_numpy(df[x_idx].values).type(torch.float)
            
            y = torch.from_numpy(df["result"].values).flatten().type(torch.float)
                                   
            s = (
                torch.from_numpy(df[data.protected_attribute_name].values)
                .flatten()
                .type(torch.float)
            )          
            
        elif self.dataset == 'compas':            
            # get train datasets
            df = data.train
            x_idx = df.columns.values.tolist()
            x_idx.remove("result")
            if self.scaling:
                x = torch.from_numpy(self.scaler.fit_transform(df[x_idx].values)).type(torch.float)
            else:
                x = torch.from_numpy(df[x_idx].values).type(torch.float)
            
            y = torch.from_numpy(df["result"].values).flatten().type(torch.float)
                                   
            s = (
                torch.from_numpy(df[data.protected_attribute_name].values)
                .flatten()
                .type(torch.float)
            )
        
        elif self.dataset == "health":
            # get train datasets
            df = data.train            
            x_idx = df.columns.values.tolist()
            x_idx.remove("MemberID")
            x_idx.remove("Year")
            x_idx.remove("CharlsonIndexI_max")
            x_idx.remove("CharlsonIndexI_min")
            x_idx.remove("CharlsonIndexI_ave")
            x_idx.remove("CharlsonIndexI_range")
            x_idx.remove("CharlsonIndexI_stdev")
            x_idx.remove("TARGET")
            x_idx.remove("trainset")
            x_idx.remove("no_PrimaryConditionGroups")
            for i in range(46):
                i += 1
                x_idx.remove(f"pcg{i}")
            for i in range(9):
                x_idx.remove(f"age_{i}5")     
            x_idx.remove("age_MISS")
            
            if self.scaling:
                x = torch.from_numpy(self.scaler.fit_transform(df[x_idx].values)).type(torch.float)
            else:
                x = torch.from_numpy(df[x_idx].values).type(torch.float)
            y = torch.from_numpy(np.array(df["CharlsonIndexI_ave"]) > 0).flatten().type(torch.float)
            s = torch.from_numpy(df[data.protected_attribute_name].values).flatten().type(torch.float)
            
        else:
            raise ValueError('only Adult, COMPAS, Health datasets are available!')

        # return tensordataset
        TD = TensorDataset(x, y, s)
        sampler = RandomSampler(TD)
        train_dataloader = DataLoader(TD, sampler=sampler, batch_size=batch_size)
               
        return train_dataloader
    
    # train dataloader
    def train(self):
        
        return self._to_dataloader(self.data, self.batch_size)
    
    # val dataloader
    def val(self):
        
        if self.dataset == 'adult':
            # get val datasets
            df = self.data.val
            x_idx = df.columns.values.tolist()
            x_idx.remove("result")
            
            if self.scaling:
                x = torch.from_numpy(self.scaler.transform(df[x_idx].values)).type(torch.float)
            else:
                x = torch.from_numpy(df[x_idx].values).type(torch.float)
            y = torch.from_numpy(df["result"].values).flatten().type(torch.float)
                        
            s = (
                torch.from_numpy(df[self.data.protected_attribute_name].values)
                .flatten()
                .type(torch.float)
            )            
            
        elif self.dataset == 'compas':
            # get val datasets
            df = self.data.val
            x_idx = df.columns.values.tolist()
            x_idx.remove("result")
            
            if self.scaling:
                x = torch.from_numpy(self.scaler.transform(df[x_idx].values)).type(torch.float)
            else:
                x = torch.from_numpy(df[x_idx].values).type(torch.float)
            y = torch.from_numpy(df["result"].values).flatten().type(torch.float)
                        
            s = (
                torch.from_numpy(df[self.data.protected_attribute_name].values)
                .flatten()
                .type(torch.float)            
                ) 
        
        elif self.dataset == "health":
            # get val datasets
            df = self.data.val
            
            x_idx = df.columns.values.tolist()
            x_idx.remove("MemberID")
            x_idx.remove("Year")
            x_idx.remove("CharlsonIndexI_max")
            x_idx.remove("CharlsonIndexI_min")
            x_idx.remove("CharlsonIndexI_ave")
            x_idx.remove("CharlsonIndexI_range")
            x_idx.remove("CharlsonIndexI_stdev")
            x_idx.remove("TARGET")
            x_idx.remove("trainset")
            x_idx.remove("no_PrimaryConditionGroups")
            for i in range(46):
                i += 1
                x_idx.remove(f"pcg{i}")
            for i in range(9):
                x_idx.remove(f"age_{i}5")
            x_idx.remove("age_MISS")
            
            if self.scaling:
                x = torch.from_numpy(self.scaler.transform(df[x_idx].values)).type(torch.float)
            else:
                x = torch.from_numpy(df[x_idx].values).type(torch.float)
            y = torch.from_numpy(np.array(df["CharlsonIndexI_ave"]) > 0).flatten().type(torch.float)
            s = torch.from_numpy(df[self.data.protected_attribute_name].values).flatten().type(torch.float) 
        else:
            raise ValueError('only Adult, COMPAS, Health datasets are available!')
        
        return x, y, s
    
    # test dataloader
    def test(self):
        
        if self.dataset == 'adult':
            # get test datasets
            df = self.data.test
            x_idx = df.columns.values.tolist()
            x_idx.remove("result")
            if self.scaling:
                x = torch.from_numpy(self.scaler.transform(df[x_idx].values)).type(torch.float)
            else:
                x = torch.from_numpy(df[x_idx].values).type(torch.float)
            y = torch.from_numpy(df["result"].values).flatten().type(torch.float)
            s = (
                torch.from_numpy(df[self.data.protected_attribute_name].values)
                .flatten()
                .type(torch.float)
            )            
            
        elif self.dataset == 'compas':
            # get test datasets
            df = self.data.test
            x_idx = df.columns.values.tolist()
            x_idx.remove("result")
            if self.scaling:
                x = torch.from_numpy(self.scaler.transform(df[x_idx].values)).type(torch.float)
            else:
                x = torch.from_numpy(df[x_idx].values).type(torch.float)
            y = torch.from_numpy(df["result"].values).flatten().type(torch.float)
            s = (
                torch.from_numpy(df[self.data.protected_attribute_name].values)
                .flatten()
                .type(torch.float)
            )            
        
        elif self.dataset == "health":
            # get val datasets
            df = self.data.test            
            x_idx = df.columns.values.tolist()
            x_idx.remove("MemberID")
            x_idx.remove("Year")
            x_idx.remove("CharlsonIndexI_max")
            x_idx.remove("CharlsonIndexI_min")
            x_idx.remove("CharlsonIndexI_ave")
            x_idx.remove("CharlsonIndexI_range")
            x_idx.remove("CharlsonIndexI_stdev")
            x_idx.remove("TARGET")
            x_idx.remove("trainset")
            x_idx.remove("no_PrimaryConditionGroups")
            for i in range(46):
                i += 1
                x_idx.remove(f"pcg{i}")
            for i in range(9):
                x_idx.remove(f"age_{i}5")   
            x_idx.remove("age_MISS")
            
            if self.scaling:
                x = torch.from_numpy(self.scaler.transform(df[x_idx].values)).type(torch.float)
            else:
                x = torch.from_numpy(df[x_idx].values).type(torch.float)
            y = torch.from_numpy(np.array(df["CharlsonIndexI_ave"]) > 0).flatten().type(torch.float)
            s = torch.from_numpy(df[self.data.protected_attribute_name].values).flatten().type(torch.float)  
        else:
            raise ValueError('only Adult, COMPAS, Health datasets are available!')
        
        return x, y, s
    
