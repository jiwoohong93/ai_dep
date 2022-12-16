import sys
import os
import yaml
import pandas as pd
import numpy as np
import librosa
import torch
from easydict import EasyDict as edict
from os.path import join
from tqdm.auto import tqdm
import shutil

from utils import extract_feature

sample_rate = 16000
window_size = 0.02
window_stride = 0.01
window = 'hamming'
def load_audio(path):
    if type(path) is str:
        sound, sample_rate = librosa.load(path, sr=16000)
    elif type(path) is tuple and len(path) == 3:
        path, start, duration = path
        sound, sample_rate = librosa.load(path, sr=16000, offset=start, duration=duration)
    # sample_rate, sound = read(path)
    # sound = sound.astype('float32') / 32767  # normalize audio
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound

def parse_audio(audio_path):

    y = load_audio(audio_path)

    n_fft = int(sample_rate * window_size)
    win_length = n_fft
    hop_length = int(sample_rate * window_stride)
    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)
    # S = log(S+1)
    spect = np.log1p(spect)
    spect = torch.FloatTensor(spect)
#     if self.normalize:
    mean = spect.mean()
    std = spect.std()
    spect.add_(-mean)
    spect.div_(std)

#     if self.spec_augment:
#         spect = spec_augment(spect)

#    return spect.numpy().T
    return np.array(spect.tolist()).T

save_dir = 'data/stft'

names = set()
os.makedirs(save_dir, exist_ok=True)
shutil.rmtree(save_dir)
os.makedirs(save_dir)

for mode in ['train', 'dev', 'test']:
    print('mode', mode)
    #df = pd.read_csv(f"data/trans_{mode}.csv")
    df = pd.read_csv(f"data/trans_{mode}_africa_american_seen_info.csv")
    for i, sample in tqdm(df.iterrows(), total=len(df)):
        if i % 5000 == 0:
            print(f'{i}th sample')
        wave = sample['file']
        speaker = sample['speaker']
        start_time, duration_time = sample.start, sample.duration
        input_file = join("data/audio", wave)
        feature = parse_audio((input_file, start_time, duration_time))
        save_name = f"{wave.replace('/','-')}-{'{:.2f}'.format(start_time)}-{'{:.2f}'.format(duration_time)}"
        save_path = join(save_dir, save_name)
        assert save_name not in names
        names.add(save_name)
        np.save(save_path, feature)
