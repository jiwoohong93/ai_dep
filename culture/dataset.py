import os
import torch

class NERPreprocessor(object):
    @staticmethod
    def preprocess(tokenizer, splited_texts_list, splited_tags_list, label_map, max_seq_len, pad_token_label_id=-100):  # bert의 최대 maxlen을 사용하는것을 
        list_of_token_ids = []
        list_of_attension_mask = []
        list_of_label_ids = []

        before_text = ' '
        for splited_texts, splited_tags in zip(splited_texts_list, splited_tags_list):
            
            tokens_for_a_sentence = []
            tags_for_a_sentence = []
            for text, tag in zip(splited_texts, splited_tags):
                if text == ' ':
                    continue
                if before_text != ' ':
                    tokens = tokenizer.tokenize('##' + text)
                else:
                    tokens = tokenizer.tokenize(text)
                tokens_for_a_sentence += tokens
                tags_for_a_sentence += list(map(lambda x: 'O' if tag == 'O' else (f'B-{tag}' if x==0 else f'I-{tag}'), range(len(tokens))))

            
            if len(tokens_for_a_sentence) > max_seq_len - 2:
                tokens_for_a_sentence = tokens_for_a_sentence[:max_seq_len - 2]
                tags_for_a_sentence = tags_for_a_sentence[:max_seq_len - 2]
            
            tokens_for_a_sentence = [tokenizer.cls_token] + tokens_for_a_sentence + [tokenizer.sep_token]
            token_ids = tokenizer.convert_tokens_to_ids(tokens_for_a_sentence)
            padding_length = max_seq_len - len(token_ids)

            attension_mask = [1]*len(token_ids) + [0]*padding_length
            token_ids += [tokenizer.pad_token_id]*padding_length

            label_ids = list(map(lambda x: label_map[x], tags_for_a_sentence))
            label_ids = [pad_token_label_id] + label_ids + [pad_token_label_id]
            label_ids += [pad_token_label_id]*padding_length

            list_of_token_ids.append(token_ids)
            list_of_attension_mask.append(attension_mask)
            list_of_label_ids.append(label_ids)
        
        return list_of_token_ids, \
            list_of_attension_mask, \
            list_of_label_ids
                        
class KLUENERPreprocessor(NERPreprocessor):
    @staticmethod
    def get_split_texts_and_tags_from_dataset_file(dataset_path: str):
        sentences = open(dataset_path, 'r').read().split("\n## ")

        splited_texts_list = []
        splited_tags_list = []
        labels = ['O']
        for sentence in sentences:
            current_tag = 'O'
            splited_texts = []
            splited_tags = []
            for idx, line in enumerate(sentence.split("\n")):
                if idx == 0 or len(line.split("\t")) < 2:
                    continue
                
                c, t = line.split("\t")
                if '-' in t:
                    target_tag = t.split("-")[-1]
                else:
                    target_tag = t

                if len(splited_texts) == 0 or target_tag != current_tag:
                    splited_texts.append(c)
                    current_tag = target_tag
                    splited_tags.append(target_tag)
                else:
                    splited_texts[-1] += c
            
            for label in splited_tags:
                if label == 'O':
                    if label not in labels:
                        labels.append(label)
                else:
                    if f'B-{label}' not in labels:
                        labels.append(f'B-{label}')
                    if f'I-{label}' not in labels:
                        labels.append(f'I-{label}')

            splited_texts_list.append(splited_texts)
            splited_tags_list.append(splited_tags)
        
        return splited_texts_list, splited_tags_list, labels

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, tokenizer, max_seq_len, labels=None, Preprocessor=None):
        assert os.path.exists(dataset_path)
        self.dataset_path = dataset_path
        self.tokenizer    = tokenizer
        self.max_seq_len  = max_seq_len

        source_text_list, target_text_list, new_labels = Preprocessor.get_split_texts_and_tags_from_dataset_file(dataset_path=dataset_path)
        assert len(source_text_list) == len(target_text_list), "Err: Number of source and target is different"
        labels = labels if labels is not None else new_labels

        label_map = {label: i for i, label in enumerate(labels)}

        list_of_token_ids, \
        list_of_attension_mask, \
        list_of_label_ids = Preprocessor.preprocess(tokenizer, source_text_list, target_text_list, label_map, max_seq_len)

        self.list_of_token_ids      = list_of_token_ids
        self.list_of_attension_mask = list_of_attension_mask
        self.list_of_label_ids      = list_of_label_ids
        self.labels                 = labels

    def __len__(self):
        return len(self.list_of_token_ids)

    def __getitem__(self, data_idx: int):
        return torch.Tensor(self.list_of_token_ids[data_idx]).long(), \
            torch.Tensor(self.list_of_attension_mask[data_idx]).long(), \
            torch.Tensor(self.list_of_label_ids[data_idx]).long()

def get_dataset(dataset_path, tokenizer, max_seq_len=50):
    if not os.path.exists(dataset_path):
        os.system(f"git clone https://github.com/KLUE-benchmark/KLUE {dataset_path}")

    dataset_path_train = os.path.join(dataset_path, "klue-ner-v1.1_train.tsv")
    dataset_path_dev = os.path.join(dataset_path, "klue-ner-v1.1_dev.tsv")
    Processor = KLUENERPreprocessor
    dataset_train = NERDataset(dataset_path_train, tokenizer=tokenizer, max_seq_len=max_seq_len, Preprocessor=Processor)
    print(f"train dataset with {len(dataset_train)} data is loaded")

    dataset_valid = NERDataset(dataset_path_dev , tokenizer=tokenizer, max_seq_len=max_seq_len, labels=dataset_train.labels, Preprocessor=Processor)
    print(f"valid dataset with {len(dataset_valid)} data is loaded")

    return dataset_train, dataset_valid