import os
import math
from pathlib import Path

import argparse
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.datasets import Dataset
from src.utils import remove_padding, entity_f1_func, char_f1_func, set_random_seed, show_random_elements

def load_dataset_local(path):
    """Load dataset from local file"""
    return torch.load(path)

def generate_datasets(task, tokenizer, data_dir, max_seq_length=512):
    train_path = os.path.join(data_dir, "klue_train.pt")
    test_path = os.path.join(data_dir, "klue_test.pt")
    
    # Try to load from local files first
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Loading datasets from local files...")
        train_data = load_dataset_local(train_path)
        test_data = load_dataset_local(test_path)
    
    else :
        print("Loading datasets from huggingface...")
        datasets = load_dataset("klue", task)
        train_data, test_data = datasets['train'], datasets['validation']
    
    LABELS = train_data.info.features['ner_tags'].feature.names
    LABEL2ID = {l:i for i,l in enumerate(LABELS)}

    train_dataset = Dataset(train_data, tokenizer, LABEL2ID, max_seq_length=max_seq_length)
    test_dataset = Dataset(test_data, tokenizer, LABEL2ID, max_seq_length=max_seq_length)
    
    return train_dataset, test_dataset, LABELS

def evaluate(model, loader, device, LABELS):
    preds, labels = predict(model, loader, device)
    preds, labels = remove_padding(preds, labels)
    entity_f1 = entity_f1_func(preds, labels, LABELS)
    char_f1 = char_f1_func(preds, labels, LABELS)
    return entity_f1, char_f1

def train_with_swa(model, loader, device, optimizer, scheduler, swa_model, swa_scheduler, epoch,
                   outside_weight=0.9, outside_idx=-1, swa_start=None):
    model.train()
    swa_model.train()

    label_weight = torch.ones(model.num_labels)
    label_weight[outside_idx] = outside_weight
    label_weight = label_weight.to(device)

    optimizer.zero_grad()  # Reset gradients

    # If starting SWA this epoch, set the constant learning rate
    if swa_start is not None and epoch == swa_start + 1:
        swa_scheduler.step()  # Set the constant SWA learning rate once

    pbar = tqdm(loader)
    for i, batch in enumerate(pbar):
        batch = [b.to(device) for b in batch]
        input_ids, attention_mask, labels = batch
        
        outputs = model(input_ids, attention_mask)
        logits = outputs.logits
        logits = logits.view(-1, model.num_labels)
        labels = labels.view(-1)

        loss = F.cross_entropy(logits, labels, weight=label_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Only update scheduler during pre-SWA phase
        if swa_start is None or epoch <= swa_start:
            scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0] if (epoch <= swa_start) else optimizer.param_groups[0]['lr']
        pbar.set_postfix({'loss': loss.item(), 'lr': current_lr})

    # Update SWA model parameters (but not learning rate) after SWA starts
    if swa_start is not None and epoch > swa_start:
        swa_model.update_parameters(model)

def predict(model, loader, device):
    model.eval()

    total_preds, total_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = [b.to(device) for b in batch]
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask, labels=labels)
            
            preds = outputs.logits.argmax(dim=-1)
            total_preds += preds.cpu().tolist()
            total_labels += labels.cpu().tolist()

    return total_preds, total_labels

def main(args):
    
    cfg = {
        'model_name': args.model,
        'device': args.device,
        'save_path': f"{args.model}_seed={args.seed}_lr={args.lr}_batch={args.batch_size}_outweight={args.outside_weight}_freeze={args.freeze}_swa",
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'outside_weight' : args.outside_weight,
        'lr': args.lr,
        # 'warmup_steps' : args.warmup_steps,
        'freeze': args.freeze,
        'max_seq_length' : args.max_seq_length,
        'swa_start': args.swa_start,
        'swa_lr': args.swa_lr,
    }

    print(cfg)

    PATH = Path(os.getcwd()).joinpath("saved", f"seed={args.seed}")
    if not os.path.exists(PATH):
        os.makedirs(PATH, exist_ok=True)

    set_random_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    model = AutoModelForTokenClassification.from_pretrained(cfg['model_name'])

    train_dataset, test_dataset, LABELS = generate_datasets(task="ner", 
                                                            tokenizer=tokenizer, 
                                                            data_dir=Path(os.getcwd()).joinpath("data/klue-ner"),
                                                            max_seq_length=args.max_seq_length)
    NUM_LABELS = len(LABELS)
    
    train_loader = train_dataset.get_dataloader(batch_size=cfg['batch_size'], shuffle=True)
    test_loader = test_dataset.get_dataloader(batch_size=cfg['batch_size'], shuffle=False)

    if model.classifier.out_features != NUM_LABELS:
        clf_in_features = model.classifier.in_features
        model.classifier = nn.Linear(clf_in_features, NUM_LABELS)
        model.num_labels = NUM_LABELS

    if cfg['freeze']:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    device = cfg['device']
    outside_weight = cfg['outside_weight']
    outside_idx = LABELS.index("O")

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    total_steps = len(train_loader) * cfg['num_epochs']

    # scheduler = get_cosine_schedule_with_warmup(optimizer,
    #     num_warmup_steps=cfg['warmup_steps'],
    #     num_training_steps=total_steps
    # )
    scheduler = CosineAnnealingLR(optimizer, 
                                  T_max=int(len(train_loader) / 5), 
                                #   T_max=int(len(train_loader) / 4), 
                                  eta_min=cfg['lr'] * 1e-1)

    # SWA setup
    swa_model = AveragedModel(model)
    swa_start = cfg['swa_start'] if cfg['swa_start'] is not None else int(0.75 * total_steps)
    swa_scheduler = SWALR(optimizer, 
                          swa_lr=cfg['swa_lr'],
                          anneal_epochs=0)

    print(f"the epoch starting swa : {swa_start}")

    ep = 0
    best_ep = -1
    init_entity_f1, init_char_f1 = evaluate(model, test_loader, device, LABELS)
    print(f'ep: {ep:02d}, entity f1: {init_entity_f1:.2f}, char f1: {init_char_f1:.2f}')

    best_score = 0.
    best_entity_f1_score = 0.

    for ep in range(1, cfg['num_epochs']+1):
        train_with_swa(model, train_loader, device, optimizer, scheduler, swa_model, swa_scheduler, ep,
                       outside_weight=outside_weight, outside_idx=outside_idx, 
                       swa_start=swa_start)
        
        if ep > cfg['swa_start']:
            # Use SWA model for evaluation
            torch.optim.swa_utils.update_bn(train_loader, swa_model)
            entity_f1, char_f1 = evaluate(swa_model, test_loader, device, LABELS)
        else:
            entity_f1, char_f1 = evaluate(model, test_loader, device, LABELS)
        
        print(f'ep: {ep:02d}, entity f1: {entity_f1:.2f}, char f1: {char_f1:.2f}')

        if char_f1 > best_score:
            save_dir = PATH.joinpath(cfg['save_path'])
            os.makedirs(save_dir, exist_ok=True)
            
            if ep > cfg['swa_start']:
                # Save full SWA model
                torch.save({
                    'model_state_dict': swa_model.module.state_dict(),
                    'model_config': swa_model.module.config,
                    'num_labels': NUM_LABELS,
                    'labels': LABELS
                }, save_dir / "pytorch_model_full.pt")
            else:
                # Save full model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_config': model.config,
                    'num_labels': NUM_LABELS,
                    'labels': LABELS
                }, save_dir / "pytorch_model_full.pt")
            
            # Save tokenizer separately
            tokenizer.save_pretrained(save_dir)
            best_score = char_f1
            best_entity_f1_score = entity_f1
            best_ep = ep + 1

    print(f'best epoch : {best_ep}, saved entity f1: {best_entity_f1_score:.2f}, saved char f1: {best_score:.2f}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="NER Training Script")
    parser.add_argument("--model", type=str, default="klue/roberta-large", help="Model name or path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    # parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--outside_weight", type=float, default=1.0, help="Outside weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--freeze", type=bool, default=False, help="Freeze all layers except classifier")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--swa_start", type=int, default=1, help="Epoch to start SWA")
    parser.add_argument("--swa_lr", type=float, default=5e-5, help="SWA learning rate")

    args = parser.parse_args()
    main(args)
