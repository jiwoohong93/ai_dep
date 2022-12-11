import torch
import spacy
from torch import nn
import numpy as np
from copy import deepcopy
from transformers import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration

from ner import *
from attribution import *


def debias_text(input_text, label, model=None):
    w_transpose = True

    if model == None:
        pretrained_weight = 't5-base'
        tokenizer = T5Tokenizer.from_pretrained(pretrained_weight)
        tokenizer.padding_side = "left" 
        tokenizer.pad_token = tokenizer.eos_token

        model = T5ForConditionalGeneration.from_pretrained(pretrained_weight)
    
    model.eval()

    learning_rate = 0.0001
    optimizer = AdamW(list(model.parameters()), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for param in model.parameters():
        param.requires_grad = True

    sample_input = tokenizer(input_text)
    sample_input_ids = torch.tensor(sample_input['input_ids'])
    sample_attention_mask = torch.tensor(sample_input['attention_mask'])

    sample_output = tokenizer(label)
    sample_label = torch.tensor(sample_output['input_ids'])

    imptGetter = ImportanceGetter(model, optimizer, criterion, tokenizer)
    layer_impts = imptGetter.get_layers_importance(sample_input_ids.unsqueeze(0), sample_attention_mask.unsqueeze(0), sample_label.unsqueeze(0), model, optimizer)
    sample_input_word_pieces_impts = imptGetter.get_word_pieces_importance(sample_input_ids.squeeze(0), layer_impts['shared.weight'])
    words_importance, clustered_words = imptGetter.cluster_word_importance(sample_input_ids, sample_input_word_pieces_impts)
    
    threshold = np.median(words_importance)
    important_words = np.array(clustered_words)[np.array(words_importance) > threshold]

    ner_model = NER()
    target_entities = ner_model.get_target_entities(important_words)
    masked_sentence = ner_model.mask_original_text(clustered_words, target_entities)
    return masked_sentence, target_entities

if __name__ == "__main__":
    input_text = 'Muslim inflicted violence on passers-by.'
    label = 'Guilty'

    masked_text, masked_entities = debias_text(input_text, label)

    print('\n----------------------')
    print(f'Original Text : {input_text}')
    print(f'Masked Tokens : {masked_entities}')
    print(f'Masked Text   : {masked_text}')
    rint('----------------------')