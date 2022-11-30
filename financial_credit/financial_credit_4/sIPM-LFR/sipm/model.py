import torch
import torch.nn as nn


class encoder_net(nn.Module):
    
    def __init__(self, num_layer, input_dim, rep_dim, acti):
        super(encoder_net, self).__init__()   
    
        if acti == "relu":
            self.acti = nn.ReLU()
        elif acti == "leakyrelu":
            self.acti = nn.LeakyReLU()
        elif acti == "softplus":
            self.acti = nn.SoftPlus()
    
        self.net = nn.ModuleList()
        for i in range(num_layer+1):
            if i == 0:
                self.net.append(nn.Linear(input_dim, rep_dim))
            else:
                self.net.append(self.acti)
                self.net.append(nn.Linear(rep_dim, rep_dim))
                
    def forward(self, x):
        for _, fc in enumerate(self.net):
            x = fc(x)
        return x
    
    
class head_net(nn.Module):
    
    def __init__(self, num_layer, input_dim, rep_dim, acti):
        super(head_net, self).__init__()   
        
        if acti == "relu":
            self.acti = nn.ReLU()
        elif acti == "leakyrelu":
            self.acti = nn.LeakyReLU()
        elif acti == "softplus":
            self.acti = nn.SoftPlus()
        elif acti == "sigmoid":
            self.acti = nn.Sigmoid()
    
        self.net = nn.ModuleList()
        for i in range(num_layer+1):
            if i == num_layer:
                self.net.append(nn.Linear(rep_dim, 1))
            else:
                self.net.append(nn.Linear(rep_dim, rep_dim))
                self.net.append(self.acti)
                
    def forward(self, x):
        for _, fc in enumerate(self.net):
            x = fc(x)
        return x, torch.sigmoid(x)
    
    
class decoder_net(nn.Module):
    
    def __init__(self, num_layer, input_dim, rep_dim, acti):
        super(decoder_net, self).__init__()    
        
        if acti == "relu":
            self.acti = nn.ReLU()
        elif acti == "leakyrelu":
            self.acti = nn.LeakyReLU()
        elif acti == "softplus":
            self.acti = nn.SoftPlus()
    
        self.net = nn.ModuleList()
        for i in range(num_layer+1):
            if i == num_layer:
                self.net.append(nn.Linear(rep_dim, input_dim))
                self.net.append(nn.Sigmoid())
            else:
                self.net.append(nn.Linear(rep_dim, rep_dim))
                self.net.append(self.acti)
                
    def forward(self, x):
        for _, fc in enumerate(self.net):
            x = fc(x)
        return x



class MLP(nn.Module):
    def __init__(self, num_layer, input_dim, rep_dim, acti):
        super(MLP, self).__init__()
        
        self.num_layer = num_layer
        self.input_dim = input_dim
        self.rep_dim = rep_dim
        self.acti = acti
        
        self.encoder = encoder_net(self.num_layer, self.input_dim, self.rep_dim, acti="leakyrelu")        
        self.head = head_net(self.num_layer, self.input_dim, self.rep_dim, self.acti)  
        self.decoder = decoder_net(self.num_layer, self.input_dim, self.rep_dim, acti="leakyrelu")  
                
    # freezing and melting
    def freeze(self):
        for para in self.parameters():
            para.requires_grad = False

    def melt(self):
        for para in self.parameters():
            para.requires_grad = True
    
    def melt_head_only(self):
        for para in self.encoder.parameters():
            para.requires_grad = False
        for para in self.decoder.parameters():
            para.requires_grad = False
        for para in self.head.parameters():
            para.requires_grad = True
            
    def replace_head(self):        
        self.head = head_net(self.num_layer, self.input_dim, self.rep_dim, self.acti)        
    
    
class MLP_linear(nn.Module):
    def __init__(self, num_layer, input_dim, rep_dim, acti):
        super(MLP_linear, self).__init__()
        
        self.num_layer = num_layer
        self.input_dim = input_dim
        self.rep_dim = rep_dim
        self.acti = acti
        
        self.encoder = encoder_net(self.num_layer, self.input_dim, self.rep_dim, acti="leakyrelu")        
        self.head = head_net(0, self.input_dim, self.rep_dim, self.acti)  
        self.decoder = decoder_net(self.num_layer, self.input_dim, self.rep_dim, acti="leakyrelu")  
                
    # freezing and melting
    def freeze(self):
        for para in self.parameters():
            para.requires_grad = False

    def melt(self):
        for para in self.parameters():
            para.requires_grad = True
    
    def melt_head_only(self):
        for para in self.encoder.parameters():
            para.requires_grad = False
        for para in self.decoder.parameters():
            para.requires_grad = False
        for para in self.head.parameters():
            para.requires_grad = True
            
    def replace_head(self):        
        self.head = head_net(self.num_layer, self.input_dim, self.rep_dim, self.acti)        
        
        
class MLP_smooth(nn.Module):
    def __init__(self, num_layer, head_num_layer, input_dim, rep_dim, acti):
        super(MLP_smooth, self).__init__()
        
        self.num_layer = num_layer
        self.head_num_layer = head_num_layer
        self.input_dim = input_dim
        self.rep_dim = rep_dim
        self.acti = acti
        
        self.encoder = encoder_net(self.num_layer, self.input_dim, self.rep_dim, acti="leakyrelu")        
        self.head = head_net(self.head_num_layer, self.input_dim, self.rep_dim, self.acti)  
        self.decoder = decoder_net(self.num_layer, self.input_dim, self.rep_dim, acti="leakyrelu")  
                
    # freezing and melting
    def freeze(self):
        for para in self.parameters():
            para.requires_grad = False

    def melt(self):
        for para in self.parameters():
            para.requires_grad = True
    
    def melt_head_only(self):
        for para in self.encoder.parameters():
            para.requires_grad = False
        for para in self.decoder.parameters():
            para.requires_grad = False
        for para in self.head.parameters():
            para.requires_grad = True
            
    def replace_head(self):        
        self.head = head_net(self.num_layer, self.input_dim, self.rep_dim, self.acti)        
    
    
class aud_Model(nn.Module):
    def __init__(self, rep_dim):
        super(aud_Model, self).__init__()
            
        # aud fc layer
        self.aud = nn.ModuleList()
        self.aud.append(nn.Linear(rep_dim, 1))
        self.aud.append(nn.Sigmoid())
            
    def forward(self, x):        
        for _, fc in enumerate(self.aud):
            x = fc(x)        
        return x
    
    def freeze(self):
        for para in self.parameters():
            para.requires_grad = False

    def melt(self):
        for para in self.parameters():
            para.requires_grad = True
            
            