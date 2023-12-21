import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import Quantizer
from scipy.cluster.vq import kmeans2


class VQVAE(nn.Module):
    def __init__(self, hp):
        super(VQVAE, self).__init__()
        self.Encoder = Encoder(hp)
        self.Quantizer = Quantizer(hp)
        self.Decoder = Decoder(hp)
        self.hp=hp
    
    def forward(self, x):
        z = self.Encoder(x)
        q, codes, quantize_loss = self.Quantizer(z)
        x = self.Decoder(q)
        return x, codes, quantize_loss
    
    def quantize(self, x):
        z = self.Encoder(x)
        _, codes, _ = self.Quantizer(z)
        return codes
    
    def decode(self, codes):
        q = self.Quantizer.q_linear(F.normalize(F.embedding(codes, self.Quantizer.codebook), p=2, dim=-1))
        x = self.Decoder(q)
        return x
    
    def init_codebook(self, data_loader):
        z_flatten = []
        with torch.no_grad():
            for i, (x, _, _) in enumerate(data_loader):
                z = self.Encoder(x.cuda(non_blocking=True))
                z_norm = F.normalize(self.Quantizer.z_linear(z), p=2, dim=-1)
                z_flatten.append( z_norm.reshape(-1, z_norm.size(-1)).data.cpu() )
        
        z_flatten = torch.cat(z_flatten, dim=0).numpy()
        print(z_flatten.shape)
        codebook = torch.from_numpy(kmeans2(z_flatten, self.hp.n_codes, minit='points')[0])
        
        self.Quantizer.codebook.data.copy_(codebook)
        return


class ResBlock(nn.Module):
    def __init__(self, hp):
        super(ResBlock, self).__init__()
        self.Layers = nn.Sequential(nn.BatchNorm1d(hp.hidden_dim),
                                    nn.ReLU(),
                                    nn.Dropout(hp.dropout),
                                    nn.Conv1d(hp.hidden_dim, hp.hidden_dim, 3, padding=1),
                                    nn.BatchNorm1d(hp.hidden_dim),
                                    nn.ReLU(),
                                    nn.Dropout(hp.dropout),
                                    nn.Conv1d(hp.hidden_dim, hp.hidden_dim, 3, padding=1))
    def forward(self, x):
        x = x.transpose(1,2)
        return (x + self.Layers(x)).transpose(1,2)
        


class Encoder(nn.Module):
    def __init__(self, hp):
        super(Encoder, self).__init__()
        self.Linear = nn.Linear(hp.orig_dim, hp.hidden_dim)
        self.stride=hp.stride
        
        if self.stride==1:
            self.RBlock1 = nn.Sequential(ResBlock(hp), ResBlock(hp), ResBlock(hp), ResBlock(hp), ResBlock(hp), ResBlock(hp))
            
        elif self.stride==2:
            self.RBlock1 = nn.Sequential(ResBlock(hp), ResBlock(hp), ResBlock(hp))
            self.Dsample1 = nn.Conv1d(hp.hidden_dim, hp.hidden_dim, 3, stride=2, padding=1)
            self.RBlock2 = nn.Sequential(ResBlock(hp), ResBlock(hp), ResBlock(hp))
            
        elif self.stride==4:
            self.RBlock1 = nn.Sequential(ResBlock(hp), ResBlock(hp))
            self.Dsample1 = nn.Conv1d(hp.hidden_dim, hp.hidden_dim, 3, stride=2, padding=1)
            self.RBlock2 = nn.Sequential(ResBlock(hp), ResBlock(hp))
            self.Dsample2 = nn.Conv1d(hp.hidden_dim, hp.hidden_dim, 3, stride=2, padding=1)
            self.RBlock3 = nn.Sequential(ResBlock(hp), ResBlock(hp))
    
    def forward(self, x):
        x = self.Linear(x)
        
        if self.stride==1:
            z = self.RBlock1(x)
            
        elif self.stride==2:
            z = self.RBlock1(x)
            z = self.Dsample1(z.transpose(1,2)).transpose(1,2)
            z = self.RBlock2(z)
        
        elif self.stride==4:
            z = self.RBlock1(x)
            z = self.Dsample1(z.transpose(1,2)).transpose(1,2)
            z = self.RBlock2(z)
            z = self.Dsample1(z.transpose(1,2)).transpose(1,2)
            z = self.RBlock3(z)
        
        return z


class Decoder(nn.Module):
    def __init__(self, hp):
        super(Decoder, self).__init__()
        self.stride=hp.stride
        
        if self.stride==1:
            self.RBlock1 = nn.Sequential(ResBlock(hp), ResBlock(hp), ResBlock(hp), ResBlock(hp), ResBlock(hp), ResBlock(hp))
            
        elif self.stride==2:
            self.RBlock1 = nn.Sequential(ResBlock(hp), ResBlock(hp), ResBlock(hp))
            self.Usample1 = nn.ConvTranspose1d(hp.hidden_dim, hp.hidden_dim, 3, stride=2, padding=1, output_padding=1)
            self.RBlock2 = nn.Sequential(ResBlock(hp), ResBlock(hp), ResBlock(hp))
            
        elif self.stride==4:
            self.RBlock1 = nn.Sequential(ResBlock(hp), ResBlock(hp))
            self.Usample1 = nn.ConvTranspose1d(hp.hidden_dim, hp.hidden_dim, 3, stride=2, padding=1, output_padding=1)
            self.RBlock2 = nn.Sequential(ResBlock(hp), ResBlock(hp), ResBlock(hp))
            self.Usample2 = nn.ConvTranspose1d(hp.hidden_dim, hp.hidden_dim, 3, stride=2, padding=1, output_padding=1)
            self.RBlock3 = nn.Sequential(ResBlock(hp), ResBlock(hp))
        
        self.Linear = nn.Linear(hp.hidden_dim, hp.orig_dim)
    
    def forward(self, x):
        if self.stride==1:
            z = self.RBlock1(x)
            
        elif self.stride==2:
            z = self.RBlock1(x)
            z = self.Usample1(z.transpose(1,2)).transpose(1,2)
            z = self.RBlock2(z)
            
        elif self.stride==4:
            z = self.RBlock1(x)
            z = self.Usample1(z.transpose(1,2)).transpose(1,2)
            z = self.RBlock2(z)
            z = self.Usample2(z.transpose(1,2)).transpose(1,2)
            z = self.RBlock3(z)
        
        z = self.Linear(z)
        
        return z
