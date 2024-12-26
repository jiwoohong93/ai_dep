import os
import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertForTokenClassification,ElectraForTokenClassification, AutoModelForTokenClassification
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from dataset import get_dataset
from models import get_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel, AutoTokenizer

import wandb


def evaluate(dataloader_valid, model, tokenizer, id2label, echo_num=40):

    total_step_per_epoch = len(dataloader_valid)
    total_loss = 0.0
    in_token_ids = None
    preds = None
    out_label_ids = None

    model.eval()
    for batch in dataloader_valid:
        batch = tuple(t.cuda() for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2]
            }

            outputs = model(**inputs)

            loss, logits = outputs[:2]
            total_loss += loss.mean().item()

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            in_token_ids = inputs["input_ids"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            in_token_ids = np.append(in_token_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0)

    loss = total_loss / total_step_per_epoch
    preds = np.argmax(preds, axis=2)

    gt_token_label_list = [[] for _ in range(out_label_ids.shape[0])]
    pred_token_label_list = [[] for _ in range(out_label_ids.shape[0])]
    gt_char_label_list = [[] for _ in range(out_label_ids.shape[0])]
    pred_char_label_list = [[] for _ in range(out_label_ids.shape[0])]

    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

    for i in range(out_label_ids.shape[0]): # sentence
        token_ids = in_token_ids[i]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        for j in range(out_label_ids.shape[1]): # token
            if out_label_ids[i, j] == pad_token_label_id: continue

            gt_label_id = out_label_ids[i][j]
            gt_label = id2label[gt_label_id]
            gt_token_label_list[i].append(gt_label)
            pred_label_id = preds[i][j]
            pred_label = id2label[pred_label_id]
            pred_token_label_list[i].append(pred_label)

            token = tokens[j]
            token = token.replace("##", "")
            if token[0] == '[' and token[-1] == ']':
                gt_char_label_list[i].append(gt_label)
                pred_char_label_list[i].append(pred_label)
            else:
                gt_char_label_list[i] += [gt_label]*len(token)
                pred_char_label_list[i] += [pred_label]*len(token)

    result = classification_report(gt_token_label_list, pred_token_label_list)
    print("[entity f1 score]")
    print(result)

    result = classification_report(gt_char_label_list, pred_char_label_list)
    print("[char f1 score]")
    print(result)
    
    return precision_score(gt_token_label_list, pred_token_label_list), \
    recall_score(gt_token_label_list, pred_token_label_list), \
    f1_score(gt_token_label_list, pred_token_label_list), \
    precision_score(gt_char_label_list, pred_char_label_list), \
    recall_score(gt_char_label_list, pred_char_label_list), \
    f1_score(gt_char_label_list, pred_char_label_list)

def main(args):

    args.wandb_name = f"KLUE-{args.model_name}-lr{args.lr}-wd{args.wd}-bs{args.bs}-ep{args.epoch}"
    args.save_path = os.path.join(args.save_path, args.wandb_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # get model
    ModelConfig, Tokenizer, Model, pretrain_name = get_model(model_name=args.model_name)

    # get tokenizer
    tokenizer = Tokenizer.from_pretrained(pretrain_name, do_lower_case=False, trust_remote_code=True)

    # get dataset
    dataset_train, dataset_valid = get_dataset(args.dataset_path, tokenizer, args.max_seq_len)
    dataloader_train = DataLoader(dataset_train, batch_size=args.bs, 
                                shuffle=True, num_workers=args.num_workers, 
                                drop_last=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.bs, shuffle=False, drop_last=False)

    labels    = dataset_train.labels
    id2label  = {i: label for i, label in enumerate(labels)}
    label2id  = {label: i for i, label in enumerate(labels)}

    config    = ModelConfig.from_pretrained(pretrain_name, num_labels=len(labels), id2label=id2label, label2id=label2id)
    model     = Model.from_pretrained(pretrain_name, config=config)
    # model = AutoModel.from_pretrained("monologg/distilkobert")
    # tokenizer = AutoTokenizer.from_pretrained("monologg/distilkobert", trust_remote_code=True)
    model.cuda()

    if Model == BertForTokenClassification:
        optimizer_grouped_parameters = [
            {'params': model.bert.parameters(), 'lr': args.lr / 100 },
            {'params': model.classifier.parameters(), 'lr': args.lr }
        ]
    elif Model == ElectraForTokenClassification:
        optimizer_grouped_parameters = [
            {'params': model.electra.parameters(), 'lr': args.lr / 100 },
            {'params': model.classifier.parameters(), 'lr': args.lr }
        ]
    elif Model == AutoModelForTokenClassification:
        optimizer_grouped_parameters = [
            {'params': model.bert.parameters(), 'lr': args.lr / 100 },
            {'params': model.classifier.parameters(), 'lr': args.lr }
        ]
    else:
        assert False, f"{Model} is not supported"

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999), eps=1e-8)
    #scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader_train) * args.epoch, eta_min=1e-6)

    echo_loss = 0.0
    best_f1score_e = 0.0
    best_f1score_c = 0.0

    total_step_per_epoch = len(dataloader_train)

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args, resume='allow', name=args.wandb_name)

    for epoch in range(args.epoch):

        # train one epoch
        total_step_per_epoch = len(dataloader_train)
        total_loss = 0.0
        echo_loss = 0.0

        model.train()
        with tqdm(dataloader_train, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                model.zero_grad()

                batch = tuple(t.cuda() for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[2]
                }
                
                outputs = model(**inputs)

                loss = outputs[0]
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                #scheduler.step()
                total_loss += loss.mean().item()
                echo_loss += loss.mean().item()

                tepoch.set_postfix(loss=loss.mean().item())
                

        loss = total_loss / total_step_per_epoch
        
        
        precision_e, recall_e, f1score_e, \
        precision_c, recall_c, f1score_c = evaluate(dataloader_valid, model, tokenizer, id2label)
        if best_f1score_c < f1score_c:
            torch.save(model.state_dict(), os.path.join(args.save_path, 'charf1_best.pth'))
            best_f1score_c = f1score_c

        if best_f1score_e < f1score_e:
            torch.save(model.state_dict(), os.path.join(args.save_path, 'entityf1_best.pth'))
            best_f1score_e = f1score_e
        stats = {}
        stats['epoch'] = epoch
        stats['train/loss']= loss
        stats['valid/entity/precision']= precision_e
        stats['valid/entity/recall'] = recall_e
        stats['valid/entity/f1-score'] = f1score_e
        stats['valid/char/precision'] =  precision_c
        stats['valid/char/recall'] =  recall_c
        stats['valid/char/f1-score'] = f1score_c

        if not args.no_wandb:
            wandb.log(stats)

        print(f"[epoch:{epoch+1}/{args.epoch}] entity train-loss:{loss:.3f}, valid-precision:{precision_e:.3f}, valid-recall:{recall_e:.3f}, valid-f1score:{f1score_e:.3f}, best-valid-f1score:{best_f1score_e:.3f}")
        print(f"[epoch:{epoch+1}/{args.epoch}] char   train-loss:{loss:.3f}, valid-precision:{precision_c:.3f}, valid-recall:{recall_c:.3f}, valid-f1score:{f1score_c:.3f}, best-valid-f1score:{best_f1score_c:.3f}")
        print()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=4, help='')

    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--save_path', type=str, default='results')

    parser.add_argument('--model_name', default='koelectra', choices=["koelectra-v3", "koelectra", "kcbert", "kobert"])
    parser.add_argument('--max_seq_len', default=50, type=int)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--wd', default=1e-2, type=float)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)

    parser.add_argument('--no_wandb', action='store_true', help='Disable WandB logging')
    parser.add_argument('--wandb_project', type=str, default='KLUE_NER', help='wandb project')
    parser.add_argument('--wandb_entity', type=str, default='jskpop', help='wandb entity')
    parser.add_argument('--wandb_name', type=str, default=None, help='wandb name')

    args = parser.parse_args()

    torch.cuda.empty_cache()
    main(args)