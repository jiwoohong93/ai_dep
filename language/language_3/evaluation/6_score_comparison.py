'''
KoBBQ
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
'''

import os
import csv
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-result-name", type=str, required=True)
    parser.add_argument("--evaluation-debiased-result-name", type=str, required=True)
    parser.add_argument('--prompt_ids', nargs='+', required=True)
    parser.add_argument('--models', nargs='+', required=True)
    args = parser.parse_args()
    return args


def load_dfs(evaluation_result_name, prompt_ids):
    df_list = []
    for prompt_id in args.prompt_ids:
        df = pd.read_csv(f'{evaluation_result_name}_{prompt_id}.tsv', sep='\t')
        df_list.append(df)
    result_df = pd.concat(df_list, axis=0)
    return result_df


def average_acc(df):
    columns_to_average = ['out-of-choice ratio', 'accuracy in ambiguous contexts', 'accuracy in disambiguated contexts', 'diff-bias in ambiguous contexts', 'diff-bias in disambiguated contexts']
    average_df = df[columns_to_average].mean()
    return average_df


def get_improvement(avg_default_df, avg_debiased_df):
    # Calculate the average of accuracy and diff-bias values for each df
    default_avg_acc_mean = (avg_default_df['accuracy in ambiguous contexts'] + avg_default_df['accuracy in disambiguated contexts']) / 2
    default_avg_diff_bias_mean = (avg_default_df['diff-bias in ambiguous contexts'] + avg_default_df['diff-bias in disambiguated contexts']) / 2

    debiased_avg_acc_mean = (avg_debiased_df['accuracy in ambiguous contexts'] + avg_debiased_df['accuracy in disambiguated contexts']) / 2
    debiased_avg_diff_bias_mean = (avg_debiased_df['diff-bias in ambiguous contexts'] + avg_debiased_df['diff-bias in disambiguated contexts']) / 2

    # Calculate the difference between the means
    accuracy_diff = debiased_avg_acc_mean - default_avg_acc_mean 
    diff_bias_diff = debiased_avg_diff_bias_mean - default_avg_diff_bias_mean

    return accuracy_diff, diff_bias_diff


def main(args):
    result_df = []
    default_df = load_dfs(args.evaluation_result_name, args.prompt_ids)
    debiased_df = load_dfs(args.evaluation_debiased_result_name, args.prompt_ids)
    for model in args.models:
        print(model)
        model_default_df = default_df[default_df['model']==model]
        model_debiased_df = debiased_df[debiased_df['model']==model]
        avg_default_df = average_acc(model_default_df)
        avg_debiased_df = average_acc(model_debiased_df)
        print('Default Average')
        print(avg_default_df)
        print('Debiased Average')
        print(avg_debiased_df)

        accuracy_diff, diff_bias_diff = get_improvement(avg_default_df, avg_debiased_df)
        print(f'diff_bias_diff: {round(diff_bias_diff*100,2)}%, accuracy_diff: {round(accuracy_diff*100,2)}%')

if __name__ == '__main__':
    args = parse_args()
    main(args)
