import os
import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertForTokenClassification,ElectraForTokenClassification
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from dataset import get_dataset
from models import get_model

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
    
    return precision_score(gt_token_label_list, pred_token_label_list), \
    recall_score(gt_token_label_list, pred_token_label_list), \
    f1_score(gt_token_label_list, pred_token_label_list), \
    precision_score(gt_char_label_list, pred_char_label_list), \
    recall_score(gt_char_label_list, pred_char_label_list), \
    f1_score(gt_char_label_list, pred_char_label_list)

def main(args):

    # get model
    ModelConfig, Tokenizer, Model, pretrain_name = get_model(model_name=args.model_name)

    # get tokenizer
    tokenizer = Tokenizer.from_pretrained(pretrain_name, do_lower_case=False)

    # get dataset
    dataset_train, dataset_valid = get_dataset(args.dataset_path, tokenizer, args.max_seq_len)
    dataloader_valid = DataLoader(dataset_valid, batch_size=args.bs, shuffle=False, drop_last=False)

    labels    = dataset_train.labels
    id2label  = {i: label for i, label in enumerate(labels)}
    label2id  = {label: i for i, label in enumerate(labels)}

    config    = ModelConfig.from_pretrained(pretrain_name, num_labels=len(labels), id2label=id2label, label2id=label2id)
    model     = Model.from_pretrained(pretrain_name, config=config, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(args.model_path))
    model.cuda()

    best_f1score_e = 0.0
    best_f1score_c = 0.0

    precision_e, recall_e, f1score_e, \
    precision_c, recall_c, f1score_c = evaluate(dataloader_valid, model, tokenizer, id2label)
    if best_f1score_c < f1score_c:
        torch.save(model.state_dict(), os.path.join(args.save_path, 'charf1_best.pth'))
        best_f1score_c = f1score_c

    if best_f1score_e < f1score_e:
        torch.save(model.state_dict(), os.path.join(args.save_path, 'entityf1_best.pth'))
        best_f1score_e = f1score_e


    print(f"entity , valid-precision:{precision_e:.3f}, valid-recall:{recall_e:.3f}, valid-f1score:{f1score_e:.3f}")
    print()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=4, help='')

    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--save_path', type=str, default='results')

    parser.add_argument('--model_name', default='koelectra', choices=["koelectra-v3", "koelectra", "kcbert"])
    parser.add_argument('--model_path', default='/workspace/AIDEP_Culture/results/KLUE-koelectra-v3-lr0.001-bs64-ep200/entityf1_best.pth', type=str)
    parser.add_argument('--max_seq_len', default=50, type=int)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    args = parser.parse_args()

    torch.cuda.empty_cache()
    main(args)