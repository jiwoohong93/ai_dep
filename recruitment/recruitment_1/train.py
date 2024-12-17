import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import random

def calculate_metrics_by_gender(y_true, y_pred, gender):
    results = {}
    for g in [0, 1]:  # 0: 남성, 1: 여성
        indices = (gender == g)
        y_true_g = y_true[indices]
        y_pred_g = y_pred[indices]
        results[g] = {
            'accuracy': accuracy_score(y_true_g, y_pred_g),
            'precision': precision_score(y_true_g, y_pred_g, zero_division=0),
            'recall': recall_score(y_true_g, y_pred_g, zero_division=0),
            'f1_score': f1_score(y_true_g, y_pred_g, zero_division=0)
        }
    return results

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    loop = tqdm(dataloader, leave=True)
    for batch in loop:
        resume_id, resume_embed, job_embed, labels = batch
        resume_embed = resume_embed.to(device); job_embed = job_embed.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        
        logits = model(resume_embed, job_embed)
        
        loss = criterion(logits.squeeze(), labels)
        
        loss.backward()
        
        optimizer.step()

        total_loss += loss.item()

        loop.set_description(f"Training")
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    
    return avg_loss


def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_idx = []

    with torch.no_grad():
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            resume_id, resume_embed, job_embed, labels = batch
            resume_embed = resume_embed.to(device); job_embed = job_embed.to(device)
            labels = labels.float().to(device)
                
            logits = model(resume_embed, job_embed)
            
            loss = criterion(logits.squeeze(), labels)
            total_loss += loss.item()

            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_idx.extend(resume_id)

            loop.set_description(f"Evaluating")
            loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]

    acc = accuracy_score(all_labels, binary_preds)
    prec = precision_score(all_labels, binary_preds, zero_division=0)
    rec = recall_score(all_labels, binary_preds, zero_division=0)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }

    total_info = {
        'resume_id': all_idx,
        'prediction': all_preds,
        'label': all_labels,
    }
    
    return metrics, total_info