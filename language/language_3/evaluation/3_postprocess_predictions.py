'''
KoBBQ
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
'''

import re
import csv
import math
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-tsv-path", type=str, required=True)
    parser.add_argument("--preprocessed-tsv-path", type=str, required=True)
    parser.add_argument("--ooc-path", type=str, default=None)
    args = parser.parse_args()
    return args


def raw2prediction(raw, choices):
    try:
        prediction = re.search('^\s*\(?(?P<raw>[^\.\n]*)\s*', raw).groupdict()['raw']
    except:
        prediction = ''
        
    prediction = prediction.replace('없습니다', '없음').replace('입니다', '')
    
    prediction_upper = prediction.upper()

    if prediction_upper and (prediction_upper[0] in choices.keys()): # starts with A, B, C
        try:
            choice = re.search('[:)]\s*(?P<choice>.*)\s*', prediction_upper).groupdict()['choice'].strip()
            
            if len(choice) == 0:
                raise Exception
            
            if choices[prediction_upper[0]] == choice.lower():
                return prediction_upper[0]
            elif sum(prediction_upper.count(choice.upper()) for choice in choices.values()) > 1: # out-of-choice
                return prediction
            else:
                # print(f"'{prediction_upper[0]}' should be '{choices[prediction_upper[0]]}', but '{prediction}' found")
                return prediction_upper[0]
            
        except:
            if sum(prediction_upper.count(alphabet) for alphabet in choices.keys()) == 1:
                prediction = prediction_upper[0]
                return prediction
        
    if prediction.lower() in choices.values(): # one of choices
        return list(choices.keys())[list(choices.values()).index(prediction.lower())]
    
    else:
        try:
            raw = re.search('\*\*[\'\"]?(?P<answer>[^\.\n\*\'\"]*)\s*', raw).groupdict()['answer']
            return raw2prediction(raw, choices)
        except:
            pass
        
        try:
            raw = re.search('답변?[은:]\s*[\'\"]?(?P<answer>[^\.\n\*\'\"]*)\s*', raw).groupdict()['answer']
            return raw2prediction(raw, choices)
        except:
            pass
        
        try:
            raw = re.search('[\'\"](?P<answer>[^\.\n\*\'\"]*)\s*', raw).groupdict()['answer']
            return raw2prediction(raw, choices)
        except:
            pass
        
        raw_upper = raw.upper()
        count = 0
        answer_alphabet = ''
        for alphabet, choice in choices.items():
            if len(re.findall(f'{alphabet}[:)]', raw_upper)) > 0:
                answer_alphabet = alphabet
                count += 1
        if count == 1:
            return answer_alphabet
        
        return raw


def postprocess(x_result, df_processed, ooc_path=None):
    x_processed = df_processed[df_processed['sample_id'] == x_result.guid]
    
    if len(x_processed) == 0:
        raise ValueError(f"{x_result.guid} Not Found")
    
    choices = {'A': x_processed.A.item().lower(), 'B': x_processed.B.item().lower(), 'C': x_processed.C.item().lower()}
    
    prediction = raw2prediction(x_result.raw, choices)
    
    if prediction not in ['A', 'B', 'C'] and ooc_path:
        with open(ooc_path, 'a', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([x_result.guid, choices, x_result.raw, prediction])
    
    return prediction


def main(args):
    tqdm.pandas()
    
    if args.ooc_path:
        Path(args.ooc_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.ooc_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['id', 'choices', 'raw', 'processed'])
    
    df_result = pd.read_csv(args.predictions_tsv_path, delimiter='\t')
    df_processed = pd.read_csv(args.preprocessed_tsv_path, delimiter='\t')

    df_result['prediction'] = df_result.progress_apply(lambda x: postprocess(x, df_processed, args.ooc_path), axis=1)
    df_result.to_csv(args.predictions_tsv_path, sep='\t', index=False)
    
    print('out-of-choice count', (~df_result['prediction'].isin(['A', 'B', 'C'])).sum())


if __name__ == '__main__':
    args = parse_args()
    main(args)
    