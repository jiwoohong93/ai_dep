import os

import numpy as np
import torch
import torch.nn as nn

from sipm.dataloader import _dataloader
from sipm.train import _train, _finetune
from sipm.eval import _eval
from sipm.model import (MLP, MLP_linear, MLP_smooth, aud_Model)
from sipm.loss import fair_Loss


class _runner:
    
    def __init__(self, dataset, scaling,
                 batch_size,
                 epochs, opt, model_lr, aud_lr, 
                 aud_steps, acti, num_layer, head_net, aud_dim,
                 eval_freq
                 ):
        
        # initialization hyps
        self.dataset = dataset
        self.scaling = bool(scaling)
        self.batch_size = batch_size
        
        # dataloaders
        dataloaders = _dataloader(dataset=self.dataset, batch_size=self.batch_size, scaling=self.scaling)
        self.train_loader = dataloaders.train()
        self.val_dataset = dataloaders.val()
        self.test_dataset = dataloaders.test()            

        # data dimensions
        self.input_dim = self.test_dataset[0].size(1)
        if self.dataset == 'adult':
            self.rep_dim = 60
        elif self.dataset == 'compas':
            self.rep_dim = 8
        elif self.dataset == 'health':
            self.rep_dim = 40
        else:
            raise NotImplementedError
        
        # learning hyps
        self.epochs = epochs
        self.opt = opt
        self.aud_opt = opt
        self.model_lr, self.aud_lr = model_lr, aud_lr            
        self.aud_steps = aud_steps
        self.acti = acti
        self.num_layer = num_layer
        self.head_net = head_net
        self.aud_num_layer = self.num_layer
        self.aud_dim = aud_dim
        self.eval_freq = eval_freq
            
        # paths
        self.config_path = f'batch-{self.batch_size}_epoch-{self.epochs}_opt-{self.opt}_lr-{self.model_lr}_advopt-{self.aud_opt}_advlr-{self.aud_lr}_advstep-{self.aud_steps}_repdim-{self.rep_dim}_head-{self.head_net}_advlayer-{self.aud_num_layer}_advdim-{self.aud_dim}/'
        self.results_path = f'results/{self.dataset}/' + self.config_path
        self.model_path = f'models/{self.dataset}/' + self.config_path        
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        
    def save_model(self, path, model, when):
        
        torch.save(model.state_dict(), os.path.join(path, f'model-{when}.pth'))
    
        
    def learning(self, i, seed, lmda, lmdaF, lmdaR):
        
        ''' fix seeds '''
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        ''' initialization '''
        if i == 0:
            if lmda > 0:
                self.model_path = self.model_path + f'sup/fair-{lmdaF}/'
            else:
                self.results_path = self.results_path + f'unsup/fair-{lmdaF}/'
            os.makedirs(self.model_path, exist_ok=True)
            os.makedirs(self.results_path, exist_ok=True)
                    
        ''' models '''
        model = MLP(num_layer=self.num_layer, 
                    input_dim=self.input_dim, rep_dim=self.rep_dim,
                    acti=self.acti
                   )
        if self.head_net == 'linear':
            model = MLP_linear(num_layer=self.num_layer, 
                    input_dim=self.input_dim, rep_dim=self.rep_dim,
                    acti=self.acti
                   )            
        elif self.head_net[0].isdigit() and (self.head_net[1:] == 'smooth'):
            model = MLP_smooth(num_layer=self.num_layer, head_num_layer=int(self.head_net[0]), 
                    input_dim=self.input_dim, rep_dim=self.rep_dim,
                    acti='sigmoid'
                   )            
        elif self.head_net[0].isdigit() and (self.head_net[1:] == 'mlp'):
            model = MLP(num_layer=self.num_layer, 
                        input_dim=self.input_dim, rep_dim=self.rep_dim,
                        acti='relu'
                       )
        else:
            raise ValueError('only linear, mlp, smooth classifiers are provided!')
        model = model.cuda()
        aud_model = aud_Model(rep_dim=self.rep_dim).cuda()
        print(model)
        print(aud_model)
        
        ''' criterion and optimizers '''
        criterion = nn.BCELoss().cuda()
        optimizer = getattr(torch.optim, self.opt)(model.parameters(), lr=self.model_lr)
        fair_criterion = fair_Loss().cuda()
        fair_optimizer = getattr(torch.optim, self.aud_opt)(aud_model.parameters(), lr=self.aud_lr)
                
        ''' train '''
        best_epoch = [0] # initial
        best_val = [-1e+10] # initial
        train_loss = []
        print('[(^◡^✿)] :::: Training ::::')
        for epoch in range(self.epochs):
            loss = _train(self.train_loader,
                          lmda, lmdaF, lmdaR,
                          model, aud_model, 
                          criterion, optimizer,
                          fair_criterion, fair_optimizer, self.aud_steps
                          )
            train_loss.append(loss)            
            # print
            print(f'[(^◡^✿)] EPOCH[{epoch+1}/{self.epochs}]: loss {loss}', end = '\r')            
            # val
            if epoch % self.eval_freq == 0:
                val_stats = _eval(self.val_dataset,
                                  model, aud_model,
                                  lmda, lmdaF, lmdaR,
                                  criterion, fair_criterion)            
                # check best
                if lmda == 0.0:
                    check = -val_stats['loss']
                else:
                    check = val_stats['acc']
                    if lmdaF > 0.0:
                        check = val_stats['acc'] - val_stats['dp']                        
                if check > best_val[-1]:
                    best_epoch.append(epoch)
                    best_val.append(check)
                    self.save_model(path=self.model_path, model=model, when='best')
                    # print
                    if lmda > 0:
                        print(f"BEST at {epoch+1} with validation | acc: {val_stats['acc']}, DP: {val_stats['dp']}")
                    else:
                        print(f"BEST at {epoch+1} with validation | loss: {val_stats['loss']}")
                            
        ''' fine tune '''
        print('[(^◡^✿)] ::: Fine-tuning :::')
        model.melt_head_only()
        for finetune_epoch in range(100):
            finetune_epoch += self.epochs
            _finetune(self.train_loader,
                      lmda, lmdaF, lmdaR,
                      model, aud_model, 
                      criterion, optimizer,
                      fair_criterion)
            finetune_val_stats = _eval(dataset=self.val_dataset, 
                                       model=model)
            # check best
            check = finetune_val_stats['acc']
            if lmdaF > 0.0:
                check = finetune_val_stats['acc'] - finetune_val_stats['dp']
            if check > best_val[-1]:
                best_epoch.append(finetune_epoch)
                best_val.append(check)
                self.save_model(path=self.model_path, model=model, when='best')
                # print
                print(f"BEST at {finetune_epoch+1} with validation | acc: {finetune_val_stats['acc']}, DP: {finetune_val_stats['dp']}")
            
    
    def inference(self, when):
        
        # inference
        best_model = MLP(num_layer=self.num_layer, 
                         input_dim=self.input_dim, 
                         rep_dim=self.rep_dim,
                         acti=self.acti
                        ).cuda()        
        if self.head_net == 'linear':
            best_model = MLP_linear(num_layer=self.num_layer, 
                                    input_dim=self.input_dim, 
                                    rep_dim=self.rep_dim,
                                    acti=self.acti
                                    ).cuda()
        elif self.head_net[0].isdigit() and (self.head_net[1:] == 'smooth'):
            best_model = MLP_smooth(num_layer=self.num_layer, 
                                    head_num_layer=int(self.head_net[0]), 
                                    input_dim=self.input_dim, 
                                    rep_dim=self.rep_dim,
                                    acti=self.acti
                                    ).cuda()
        elif self.head_net[0].isdigit() and (self.head_net[1:] == 'mlp'):
            best_model = MLP_smooth(num_layer=self.num_layer, 
                                    head_num_layer=int(self.head_net[0]),
                                    input_dim=self.input_dim, 
                                    rep_dim=self.rep_dim,
                                    acti='relu'
                                    ).cuda()
        else:
            raise ValueError('only linear, mlp, smooth classifiers are provided!')        
        best_model.load_state_dict(torch.load(os.path.join(self.model_path, f'model-{when}.pth')))
        best_model.eval()        
        test_stats = _eval(dataset=self.test_dataset,
                           model=best_model)        
        print('BEST test results:')
        print(test_stats)
        
        return test_stats


        
        