import re

import nltk
nltk.download('punkt')

import pandas as pd
from torch.utils.data import Dataset

class MyDataset(Dataset):
  def __init__(self, tokenizer, path='./filtered_data.csv', outout=None, max_len=200):
    self.df = pd.read_csv(path,sep='_')
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []

    self._build()
  
  def __len__(self):
    return len(self.inputs)
  
  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()

    src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
    target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
  
  def _build(self):
    self._buil_examples_from_files()
  
  def _buil_examples_from_files(self,):
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    for row in self.df.iterrows():
        text = row[1]['text']
        sentiment = row[1]['label'] if row[1]['label'] == 'negative' else 'positive'

        
        line = text.strip()
        line = REPLACE_NO_SPACE.sub("", line) 
        line = REPLACE_WITH_SPACE.sub("", line)
        line = line + ' </s>'

        target = sentiment + " </s>"

        # tokenize inputs
        tokenized_inputs = self.tokenizer.batch_encode_plus(
            [line], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
        )
        # tokenize targets
        tokenized_targets = self.tokenizer.batch_encode_plus(
            [target], max_length=2, pad_to_max_length=True, return_tensors="pt"
        )

        self.inputs.append(tokenized_inputs)
        self.targets.append(tokenized_targets)