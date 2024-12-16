import torch
import os
import pickle
import json


def save_pkl(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    return

def load_pkl(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def read_txt(file):
    file = open(file, 'r')
    data = []
    while True:
        line = file.readline()
        data.append(line.strip().split('\t'))
        if not line:
            break
    file.close()

    return data

def dict_to_device(dict, device):
    for key in dict.keys():
        dict[key] = dict[key].to(device)
    return dict

def dict_to_devices(dict, rank):
    for key in dict.keys():
        dict[key] = dict[key].cuda(rank)
    return dict

def compute_metrics(logits, labels, processor, wer_metric):
	pred_ids = torch.argmax(logits, axis=-1)
	labels[labels == -100] = processor.tokenizer.pad_token_id
	pred_str = processor.batch_decode(pred_ids)
	# we do not want to group tokens when computing the metrics
	label_str = processor.batch_decode(labels, group_tokens=False)
	wer = wer_metric.compute(predictions=pred_str, references=label_str)
	return {"wer": wer}