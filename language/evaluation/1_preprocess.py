'''
KoBBQ
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
'''

import json
import scipy
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-tsv-path", type=str, required=True)
    parser.add_argument("--evaluation-tsv-path", type=str, default=None)
    parser.add_argument("--evaluation-json-path", type=str, default=None)
    parser.add_argument("--prompt-tsv-path", type=str, required=True)
    parser.add_argument("--prompt-id", type=int, required=True)
    args = parser.parse_args()
    return args


def change_unk(x, unk):
    if x == '알 수 없음':
        return unk
    else:
        return x
        

def change_unk_choice(x, unk):
    return [change_unk(choice, unk) for choice in x]


def permute(df):
    processed_df = pd.DataFrame(np.repeat(df.values, 3, axis=0), columns=df.columns)
    permute_choices = pd.DataFrame(
        np.concatenate(df['choices'].apply(lambda x: scipy.linalg.circulant(x))),
        columns=['A', 'B', 'C']
    )
    processed_df = pd.concat([processed_df, permute_choices], axis=1)
    processed_df['sample_id'] = processed_df['sample_id'] + ['-0', '-1', '-2'] * len(df)
    processed_df['choices'] = processed_df.apply(lambda x: [x.A, x.B, x.C], axis=1)
    return processed_df


def make_prefix(df_prompts, prompt_id):
    prefix = df_prompts[df_prompts['prompt_id'] == prompt_id]['instruction'].item()
    return prefix + '\n\n'
        

def make_prompt(df_prompts, prompt_id, context, question, A, B, C):
    prompt_row = df_prompts[df_prompts['prompt_id'] == prompt_id]
    
    prompt = prompt_row['context'].item() + context + '\n'
    prompt += prompt_row['question'].item() + question + '\n'
    prompt += prompt_row['a'].item() + A + '\n'
    prompt += prompt_row['b'].item() + B + '\n'
    prompt += prompt_row['c'].item() + C + '\n'
    prompt += prompt_row['answer']
    
    return prompt


def process(df, df_prompts, prompt_id):
    unk = df_prompts[df_prompts['prompt_id'] == prompt_id]['unknown'].item()
    
    df['choices'] = df['choices'].apply(lambda x: change_unk_choice(x, unk))
    df['answer'] = df['answer'].apply(lambda x: change_unk(x, unk))

    df = permute(df)
    df['answer_abc'] = df[['answer', 'A', 'B', 'C']].apply(lambda x: x[x == x.answer].index.tolist()[-1], axis=1)
        
    prefix = make_prefix(df_prompts, prompt_id)
    df['query'] = df.apply(lambda x: make_prompt(df_prompts, prompt_id, x.context, x.question, x.A, x.B, x.C), axis=1)

    df = df.sort_values(by=['sample_id'])
    return df, prefix


def to_tsv(df, tsv_path):
    Path(tsv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tsv_path, sep='\t', index=False)
    

def to_json(df, prefix, json_path):
    data = {
        "ver": "test",
        "prefix": prefix,
        "data": df[['query', 'A', 'B', 'C', 'answer_abc', 'sample_id']].values.tolist()
    }
    
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main(args):
    df = pd.read_csv(args.samples_tsv_path, sep='\t')
    df['choices'] = df['choices'].map(eval)
    
    df_prompts = pd.read_csv(args.prompt_tsv_path, sep='\t')
    
    df, prefix = process(df, df_prompts, args.prompt_id)
    
    if args.evaluation_json_path:
        to_json(
            df,
            prefix,
            args.evaluation_json_path
        )
    if args.evaluation_tsv_path:
        df = df.drop('query', axis=1)
        to_tsv(df, args.evaluation_tsv_path)

    
if __name__ == '__main__':
    args = parse_args()
    main(args)
    
