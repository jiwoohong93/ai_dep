'''
KoBBQ
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
'''

import csv
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-tsv-path", type=str, required=True)
    parser.add_argument("--preprocessed-tsv-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    return args


def abc_2_prediction(x_processed, df_result):
    pred = df_result[df_result.guid == x_processed.sample_id].prediction
    
    if len(pred) == 1:
        pred = pred.item()
    elif len(pred) == 0:
        raise ValueError(f"{x_processed.sample_id} Not Found")
    else:
        raise ValueError(f"{len(pred)} {x_processed.sample_id} Found")

    if pred in ['A', 'B', 'C']:
        pred = x_processed[pred]

    return pred


def to_model_evaluation_tsv(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False)


def main(args):
    tqdm.pandas()
    
    df_result = pd.read_csv(args.predictions_tsv_path, delimiter='\t')
    df_evaluation = pd.read_csv(args.preprocessed_tsv_path, delimiter='\t')

    df_evaluation['prediction'] = df_evaluation.progress_apply(lambda x: abc_2_prediction(x, df_result), axis=1)

    to_model_evaluation_tsv(df_evaluation, args.output_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)