import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataset import load_data
from preprocess import load_resources, clean_text

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_key, label_key, bert_tokenizer):
        self.sentences = [bert_tokenizer(str(text), truncation=True, padding='max_length', return_tensors='pt') for text in dataset[sent_key].tolist()]
        self.labels = [np.int64(i) for i in dataset[label_key]] if label_key else [np.int64(0) for i in dataset[sent_key]]
        self.mode = "train" if label_key else "test"

    def __getitem__(self, i):
        item = {key: val.squeeze() for key, val in self.sentences[i].items()}
        if self.mode == "train":
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def collate_fn(batch):
    collated_data = {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
    }
    
    if 'labels' in batch[0]:
        collated_data['labels'] = torch.stack([item['labels'] for item in batch])
    
    return collated_data
    
def main(model_path, test):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    label_col = 'section'
    data_test = BERTDataset(test, "title", None, tokenizer)
    test_loader_bert = DataLoader(data_test, batch_size=32, collate_fn=collate_fn)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []
    probabilities = []

    with torch.no_grad():
        for batch in test_loader_bert:
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            predictions.extend(preds)
            probabilities.extend(probs)

    test['pred'] = predictions
    test['pred_probs'] = probabilities

    y_true = test[label_col].values
    y_pred = predictions

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)

    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    train, test = load_data()
    resources = load_resources()

    train['title'] = train['title'].apply(lambda x: clean_text(x, resources['stopwords'], resources['cn_to_ko'], resources['symbol_replace'], resources['regex_patterns'], resources['people']))
    test['title'] = test['title'].apply(lambda x: clean_text(x, resources['stopwords'], resources['cn_to_ko'], resources['symbol_replace'], resources['regex_patterns'], resources['people']))
    print(test.columns)
    print(len(test))
    model_path = "./best_model"
    main(model_path, test)