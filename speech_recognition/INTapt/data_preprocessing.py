import os
import sys 

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(parent_dir)

import torch
import glob
import ipdb
import argparse
import soundfile as sf
import librosa
import re
import string

from datasets import load_dataset, load_metric, concatenate_datasets
from datasets import Dataset, Audio

import utils

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def down_sample(wave_file, input_sample_rate, output_sample_rate):
    orig_wave, sample_rate = librosa.load(wave_file, sr=input_sample_rate)
    resampled_wave = librosa.resample(orig_wave, sample_rate, output_sample_rate)
    return resampled_wave

def main():
    parser = argparse.ArgumentParser(description='Preprocessing Coraal dataset')

    parser.add_argument('--data_dir', type=str, default='data/coraal')
    parser.add_argumnet('--save_per_accents', action='store_true')
    
    args = parser.parse_args()
    

    speakers = ['ATL', 'DCA', 'DCB', 'LES', 'PRV', 'ROC', 'VLD']

    total_train_dataset = {}
    total_test_dataset = {}

    total_train_dataset = Dataset.from_dict(total_train_dataset)
    total_test_dataset = Dataset.from_dict(total_test_dataset)

    for speaker in speakers:
        total_dataset = {}

        total_dataset = Dataset.from_dict(total_dataset)

        audio_path = os.path.join(args.data_dir, speaker + '_audio')
        text_path = os.path.join(args.data_dir, speaker + '_texts')

        file_list = [file[:-4] for file in os.listdir(text_path)]

        audio_files = glob.glob(os.path.join(audio_path, '*'))
        texts = glob.glob(os.path.join(text_path, '*'))

        texts = []
        audios = []
        speakers = []
        orig_sr = 44100
        target_sr = 16000
        file_list = [file for file in file_list if '.' not in file]
        
        for file in file_list:
            audio_file = os.path.join(audio_path,file+'.wav')
            text_file = os.path.join(text_path, file+'.txt')
            text_data = utils.read_txt(text_file)[1:-1]
            
            resampled_audio = down_sample(audio_file, orig_sr, target_sr)

            for idx, line in enumerate(text_data):
                if line[1] == file[:-2]:
                    audio = resampled_audio[int(float(line[2])*target_sr): int(float(line[4])*target_sr)]

                    text = re.sub("[\(\<\-\/].*?[\)\>\-\/]", "", line[3])
                    if text == line[3]:
                        text = text.translate(str.maketrans('', '', string.punctuation))

                        if len(text.split()) > 2:
                            text = line[3].upper()
                            audio_dict = {'array': audio, 'sampling_rate':target_sr}
                            audios.append(audio_dict)
                            texts.append(text)
                            speakers.append(line[1])
                    else:
                        continue

        total_dataset = Dataset.from_dict({'audio':audios})
        total_dataset = total_dataset.add_column(name='speaker', column=speakers)
        total_dataset = total_dataset.add_column(name='text', column=texts)
        dataset_path = os.path.join(args.data_dir, speaker + '.hf')
        if args.save_per_accents:
            total_dataset.save_to_disk(dataset_path)

        accents = [speaker]*len(total_dataset)
        gender = [speaker.split('_')[-2] for speaker in total_dataset['speaker']]
        age = [speaker.split('_')[-3] for speaker in total_dataset['speaker']]
        total_dataset = total_dataset.add_column('accent', accents)
        total_dataset = total_dataset.add_column('gender', gender)
        total_dataset = total_dataset.add_column('age', age)

        total_dataset = total_dataset.filter(lambda x: '[' not in x['text'] or ']' not in x['text'])

        total_dataset = total_dataset.train_test_split(test_size=0.1)
        train_dataset = total_dataset['train']
        test_dataset = total_dataset['test']

        total_train_dataset = concatenate_datasets([total_train_dataset, train_dataset])
        total_test_dataset = concatenate_datasets([total_test_dataset, test_dataset])

    total_train_dataset.save_to_disk(os.path.join(args.data_dir, 'clean_train.hf'))
    total_test_dataset.save_to_disk(os.path.join(args.data_dir, 'clean_test.hf'))

