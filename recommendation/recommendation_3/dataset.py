import pandas as pd
from datasets import load_dataset

def load_data():
    dataset = load_dataset('klue/klue', 'ynat')

    klue_train = pd.DataFrame({
        'title': dataset['train']['title'],
        'label': dataset['train']['label'],
        'url': dataset['train']['url'],
        'date': dataset['train']['date']
    })
    klue_test = pd.DataFrame({
        'title': dataset['validation']['title'],
        'label': dataset['validation']['label'],
        'url': dataset['validation']['url'],
        'date': dataset['validation']['date']
    })

    klue_train = klue_train[(klue_train['label'] == 6) | (klue_train['label'] == 1) | (klue_train['label'] == 2)]
    klue_test = klue_test[(klue_test['label'] == 6) | (klue_test['label'] == 1) | (klue_test['label'] == 2)]
    klue_train['label'] = klue_train['label'].replace({6: 0, 2: 1, 1: 2})
    klue_test['label'] = klue_test['label'].replace({6: 0, 2: 1, 1: 2})
    klue_train.rename(columns={'label': 'section'}, inplace=True)
    klue_test.rename(columns={'label': 'section'}, inplace=True)
    return klue_train, klue_test