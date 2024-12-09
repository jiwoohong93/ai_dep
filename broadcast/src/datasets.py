import torch
from src.utils import pad_seqs
from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, LABEL2ID, max_seq_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.outside_label = LABEL2ID['O']
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = ''.join(item['tokens'])
        char_labels = item['ner_tags']

        inputs = self.tokenizer(text)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        labels = [self.outside_label] * len(input_ids)
        for i in range(len(input_ids)):
            span = inputs.token_to_chars(i)
            if span is not None:
                labels[i] = char_labels[span.start]
        
        return input_ids, attention_mask, labels
    
    def collate_fn(self, batch):
        input_ids, attention_mask, labels = zip(*batch)
        input_ids = pad_seqs(input_ids, self.pad_token_id, max_length=self.max_seq_length)
        attention_mask = pad_seqs(attention_mask, 0, max_length=self.max_seq_length)
        labels = pad_seqs(labels, -100, max_length=self.max_seq_length)
        return input_ids, attention_mask, labels
    
    def get_dataloader(self, batch_size, shuffle):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
    

class CollateNer(object):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, label2idx: Dict[str, int], max_length: int
    ):
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        self.max_length = max_length

    def __call__(self, input_examples):
        input_texts, input_labels_str = [], []
        for input_example in input_examples:
            text, label_strs = input_example
            input_texts.append(text)
            input_labels_str.append(label_strs)

        encoded_texts = self.tokenizer.batch_encode_plus(
            input_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",  # KLUE 베이스라인과 input형태를 일치 시키기 위해
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        input_ids = encoded_texts["input_ids"]
        token_type_ids = encoded_texts["token_type_ids"]
        attention_mask = encoded_texts["attention_mask"]

        len_input = input_ids.size(1)
        input_labels = []
        for input_label_str in input_labels_str:
            input_label_str = (
                ["O"] + input_label_str + (len_input - len(input_label_str) - 1) * ["O"]
            )
            input_label = [self.label2idx[x] for x in input_label_str]
            input_label = torch.tensor(input_label).long()
            input_labels.append(input_label)

        input_labels = torch.stack(input_labels)
        return input_ids, token_type_ids, attention_mask, input_labels


class NerDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset: List[Dict[str, Union[str, List[str]]]],
        label_list: List[str],
        max_length: int,
        batch_size: int = None,
        shuffle: bool = False,
        **kwargs
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.label2idx = {label: i for i, label in enumerate(label_list)}
        self.collate_fn = CollateNer(tokenizer, self.label2idx, max_length)
        self.loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            **kwargs
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        instance = self.dataset[index]
        text = instance["text_a"]
        label_strs = instance["label"]

        return text, label_strs
