
import torch
import itertools
from sklearn.metrics import f1_score
from seqeval.metrics import f1_score as ner_f1_score
from seqeval.scheme import IOB2

from datasets import ClassLabel
from IPython.display import display, HTML
import random
import pandas as pd
import numpy as np

def set_random_seed(random_seed) :
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def pad_seqs(seqs, pad_val, max_length=256):
    _max_length = max([len(s) for s in seqs])
    max_length = _max_length if max_length is None else min(_max_length, max_length)
    
    padded_seqs = []
    for seq in seqs:
        seq = seq[:max_length]
        pads = [pad_val] * (max_length - len(seq))
        padded_seqs.append(seq + pads)
    
    return torch.tensor(padded_seqs)

def remove_padding(preds, labels):
    removed_preds, removed_labels = [], []
    for p, l in zip(preds, labels):
        if -100 not in l: continue

        idx = l.index(-100)
        removed_preds.append(p[:idx])
        removed_labels.append(l[:idx])
    
    return removed_preds, removed_labels

def entity_f1_func(preds, targets, LABELS):
    preds = [[LABELS[p] for p in pred] for pred in preds]
    targets = [[LABELS[t] for t in target] for target in targets]
    entity_macro_f1 = ner_f1_score(targets, preds, average="macro", mode="strict", scheme=IOB2)
    f1 = entity_macro_f1 * 100.0
    return round(f1, 2)

def char_f1_func(preds, targets, LABELS):
    label_indices = list(range(len(LABELS)))
    preds = list(itertools.chain(*preds))
    targets = list(itertools.chain(*targets))
    f1 = f1_score(targets, preds, labels=label_indices, average='macro', zero_division=True) * 100.0
    return round(f1, 2)

def show_random_elements(dataset, num_examples=30):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."

    picks = []
    
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)

        while pick in picks:
            pick = random.randint(0, len(dataset)-1)

        picks.append(pick)

    df = pd.DataFrame(dataset[picks])

    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])

    display(HTML(df.to_html()))