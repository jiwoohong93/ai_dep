import os
import yaml
import argparse
import logging
import shutil
import librosa
import numpy as np
import torch

from os.path import join
from easydict import EasyDict as edict

def save_checkpoint(model, optimizer, performance, epoch, save_dir='.', head=None):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'performance': performance,
    }
    if save_dir != '.':
        os.makedirs(save_dir, exist_ok=True)
    filename = f'{head}.pt'
    torch.save(state, join(save_dir, filename))


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pad_list(xs, pad_value):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


def extract_feature(args, input_file, add_noise=True, **kwargs):
    start = kwargs.get('start', None)
    duration = kwargs.get('duration', None)
    input_dim = args.feature.input_dim
    window_size = args.feature.window_size
    stride = args.feature.stride
    cmvn = args.feature.cmvn

    if start is not None and duration is not None:
        y, sr = librosa.load(input_file, sr=8000, offset=start, duration=duration)
    else:
        y, sr = librosa.load(input_file, sr=8000)

    if add_noise:
        noise = np.random.randn(len(y))
        y = y + 0.01 * noise
    ws = int(sr * 0.001 * window_size)
    st = int(sr * 0.001 * stride)
    feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=input_dim, n_fft=ws, hop_length=st)
    feat = np.log(feat + 1e-6)

    feat = [feat]

    feat = np.concatenate(feat, axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]

    return np.swapaxes(feat, 0, 1).astype('float32')
