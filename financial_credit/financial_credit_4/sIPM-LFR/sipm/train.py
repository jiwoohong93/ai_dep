import torch


def _loss(x, y, s,
          lmda, lmdaF, lmdaR,
          model, aud_model,
          criterion, fair_criterion):
        
        # to train
        model.train()
        aud_model.train()
        
        # weights and flattening
        y, s = y.flatten(), s.int().flatten()
        
        # feeding
        z = model.encoder(x)
        _, preds = model.head(z)
        
        # task loss
        task_loss = criterion(preds.flatten(), y)
        
        # fair loss
        fair_loss = 0.0
        if lmdaF > 0.0:
            z0, z1 = z[s == 0], z[s == 1]
            aud_z0, aud_z1 = aud_model(z0), aud_model(z1)
            fair_loss = fair_criterion(aud_z0, aud_z1)            
            
        # recon loss
        recon_loss = 0.0
        if lmdaR > 0.0:
            recon = model.decoder(z)
            recon_loss = ((x - recon)**2).sum(dim=1).mean()
        
        # all loss
        loss = lmda*task_loss
        if lmdaF > 0.0:
            loss += lmdaF*fair_loss
        if lmdaR > 0.0:
            loss += lmdaR*recon_loss
        
        return loss
        

def _train(train_loader,
           lmda, lmdaF, lmdaR,
           model, aud_model, 
           criterion, optimizer,
           fair_criterion, fair_optimizer, aud_steps
          ):
    
    losses = 0.0
    n_train = 0
    
    for x, y, s in train_loader:
        
        # initialization
        batch_size = x.size(0)
        n_train += batch_size
        x, y, s = x.cuda(), y.cuda(), s.cuda()
        
        # train encoder + head
        model.melt()
        aud_model.freeze()
        
        loss = _loss(x, y, s, 
                     lmda, lmdaF, lmdaR, 
                     model, aud_model, 
                     criterion, fair_criterion
                     )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) # technical
        optimizer.step()
        losses += loss.item() * batch_size
        
        # train adversarial network
        if lmdaF > 0:
            model.freeze()
            aud_model.melt()
            for _ in range(aud_steps):
                loss = _loss(x, y, s, 
                             lmda, lmdaF, lmdaR, 
                             model, aud_model, 
                             criterion, fair_criterion)
                loss *= -1
                fair_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(aud_model.parameters(), 5.0)
                fair_optimizer.step()
        
    return round(losses / n_train, 5)
            
    
def _finetune(train_loader,
              lmda, lmdaF, lmdaR,
              model, aud_model,
              criterion, optimizer,
              fair_criterion
             ):

    # only lmda = 1.0
    lmda, lmdaF, lmdaR = 1.0, 0.0, 0.0

    xs, zs, ys, ss = [], [], [], []
    for x, y, s in train_loader:
        # initialization
        x, y, s = x.cuda(), y.cuda(), s.cuda()
        # train encoder + head
        model.melt_head_only()
        aud_model.freeze()
        loss = _loss(x, y, s, 
                     lmda, lmdaF, lmdaR, 
                     model, aud_model, 
                     criterion, fair_criterion)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    

        