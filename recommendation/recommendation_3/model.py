import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_key, label_key, bert_tokenizer, max_length=128):
        self.sentences = [
            bert_tokenizer(str(text), truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
            for text in dataset[sent_key].tolist()
        ]
        self.labels = [np.int64(i) for i in dataset[label_key]] if label_key else [np.int64(0) for i in dataset[sent_key]]
        self.mode = "train" if label_key else "test"

    def __getitem__(self, i):
        item = {key: val.squeeze() for key, val in self.sentences[i].items()}
        if self.mode == "train":
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def load_tokenizer_and_model(model_checkpoint, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels, ignore_mismatched_sizes=True)
    return tokenizer, model