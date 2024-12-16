import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score

from model import BERTDataset, load_tokenizer_and_model
from metrics import compute_metrics
from hp_search import run_hp_search
from transformers import Trainer, TrainingArguments, set_seed
import random
import numpy as np
import torch
from dataset import load_data
from preprocess import load_resources, clean_text

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def train_model(train_df, val_df, test_df, model_checkpoint, num_labels, label_col, output_dir, random_seed=42):
    set_random_seed(random_seed)
    tokenizer, model = load_tokenizer_and_model(model_checkpoint, num_labels)
    data_train = BERTDataset(train_df, "title", label_col, tokenizer)
    data_val = BERTDataset(val_df, "title", label_col, tokenizer)
    data_test = BERTDataset(test_df, "title", None, tokenizer)

    def model_init():
        return model

    best_trial = run_hp_search(data_train, data_val, tokenizer, compute_metrics, model_init, output_dir)
    print("Best Trial:")
    print(f"  Trial Number: {best_trial.number}")
    print(f"  Parameters: {best_trial.params}")
    print(f"  Accuracy: {best_trial.value}")
    print(f"  Additional Info: {best_trial.user_attrs}")

    best_learning_rate = best_trial.params["learning_rate"]
    best_batch_size = best_trial.params["batch_size"]
    
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=100,
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        learning_rate=best_learning_rate,
        per_device_train_batch_size=best_batch_size,
        per_device_eval_batch_size=best_batch_size,
        num_train_epochs=10,
        save_total_limit=2
    )
    
    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=data_train,
        eval_dataset=data_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    
    if trainer.state.best_model_checkpoint:
        print(f"Training stopped early at checkpoint: {trainer.state.best_model_checkpoint}")
        best_model_dir = os.path.join(output_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        model = model.from_pretrained(trainer.state.best_model_checkpoint)
        model.save_pretrained(best_model_dir)
        trainer.save_model(best_model_dir)  
        trainer.state.save_to_json(os.path.join(best_model_dir, "trainer_state.json"))
    else:
        print("No early stopping occurred, saving the final model.")
        trainer.save_model(output_dir)


    pred = trainer.predict(data_test)
    logits = pred.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    test_df['pred'] = np.argmax(probs, axis=1)
    test_df['pred_probs'] = probs.tolist()
    
    y_true = test_df[label_col].values
    y_pred = test_df['pred'].values

    report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(num_labels)])
    print("[Classification Report]")
    print(report)

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy


def trainer(filepath, train, test, model_checkpoint, model_name, random_seed=42):
    set_random_seed(random_seed)
    label_col = 'section'
    train_df, val_df = train_test_split(train, test_size=0.2, random_state=random_seed, stratify=train[label_col])

    current_date = datetime.now().strftime("%m%d")
    result_path = os.path.join(filepath, f"result_{model_name}_{current_date}")
    os.makedirs(result_path, exist_ok=True)

    precision, recall, f1, accuracy = train_model(train_df, val_df, test, model_checkpoint, 3, label_col, result_path)
    print(f"precision:{precision:.3f}, recall:{recall:.3f}, f1score:{f1:.3f}, accuracy:{accuracy:.3f}")


if __name__ == "__main__":
    FILEPATH = "./model"
    train, test = load_data()
    resources = load_resources()

    train['title'] = train['title'].apply(lambda x: clean_text(x, resources['stopwords'], resources['cn_to_ko'], resources['symbol_replace'], resources['regex_patterns'], resources['people']))
    test['title'] = test['title'].apply(lambda x: clean_text(x, resources['stopwords'], resources['cn_to_ko'], resources['symbol_replace'], resources['regex_patterns'], resources['people']))
    
    MODEL_CHECKPOINT = "klue/bert-base"
    MODEL_NAME = "bert"

    trainer(FILEPATH, train, test, MODEL_CHECKPOINT, MODEL_NAME)