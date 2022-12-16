# Original Code: https://github.com/Hertin/Equal-Accuracy-Ratio

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import os
import sys
import argparse
import logging
import yaml

from models.rnnctc import RNNCTC
from models.deepspeech2 import DeepSpeech2
from data import *

from utils import AverageMeter, clip_gradient
from ctcdecode import CTCBeamDecoder
from Levenshtein import distance as levenshtein_distance

def get_logger(logfile_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', '%Y-%m-%d %H:%M:%S')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(logfile_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
    
def check_loss(loss, loss_value):
    """
    Check that warp-ctc loss is valid and will not break training
    :return: Return if loss is valid, and the error in case it is not
    """
    loss_valid = True
    error = ''
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        error = "WARNING: received an inf loss"
    elif torch.isnan(loss).sum() > 0:
        loss_valid = False
        error = 'WARNING: received a nan loss, setting loss value to 0'
    elif loss_value < 0:
        loss_valid = False
        error = "WARNING: received a negative loss"
    return loss_valid, error

def save_model(conf, model, name):
    checkpoint_dir = os.path.join(conf.saved_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{name}.pt")
    torch.save(model.state_dict(), checkpoint_path)

def train_one_epoch(conf, data_loader, model, optimizer, cur_epoch, print_freq, saved_freq, saved_dir, lr):
    # switch to train mode
    model.train()

    losses = AverageMeter()

    for i, (index, attr, features, trns, input_lengths) in enumerate(data_loader):
        # Move to GPU, if available
        if features.size(1) > 2500:
            conf.logger.info(f'feature too long discard, length = {features.size(i)}')
            continue
        features = features.float().cuda()
        trns = trns.long().cuda()
        input_lengths = input_lengths.int()
        if conf.model_name == "rnnctc":
            input_lengths = input_lengths.cuda()

        # Forward prop.
        raw_loss = model(features, input_lengths, trns)
        loss_value = raw_loss.mean().item()
        valid_loss, error = check_loss(raw_loss, loss_value)

        if valid_loss:
            optimizer.zero_grad()
            loss = raw_loss.mean()
            with torch.no_grad():
                losses.update(loss.item())

            loss.backward()
            clip_gradient(optimizer, conf.grad_clip)
            optimizer.step()
        else:
            conf.logger.info(error)
            conf.logger.info('Skipping grad update')

        if i % conf.print_freq == 0:
            conf.logger.info(f"Epoch {cur_epoch} | Batch {i} / {len(data_loader)} | Loss {losses.val:.4f} ({losses.avg:.4f})")

    return losses


def validate(conf, model, epoch, dev_loader, device, label_set):
    # evaluation mode
    model.eval()

    losses = AverageMeter()

    total_error = 0.
    total_length = 0.
    IVOCAB = {i: l for i, l in enumerate(label_set)}
    decoder = CTCBeamDecoder(labels=list(IVOCAB.values()), beam_width=conf.beam_width, log_probs_input=True)

    with torch.no_grad():
        for i, (index, attr, features, trns, input_lengths) in enumerate(dev_loader):
            if features.size(1) > 2500:
                conf.logger.info(f'feature too long discard, length = {features.size(i)}')
                continue

            features = features.float().to(device)
            trns = trns.long().to(device)
            input_lengths = input_lengths.int()
            if conf.model_name == "rnnctc":
                input_lengths = input_lengths.to(device)

            raw_loss, logit = model(features, input_lengths, trns, logit=True)
            out, scores, offsets, seq_lens = decoder.decode(logit, model.get_seq_lens(input_lengths))

            for hyp, trn, length in zip(out, trns, seq_lens):
                best_hyp = hyp[0,:length[0]]
                best_hyp_str = ''.join(list(map(chr, best_hyp)))
                t = trn.detach().cpu().tolist()
                t = [ll for ll in t if ll != 0]
                tlength = len(t)
                truth_str = ''.join(list(map(chr, t)))

                error = levenshtein_distance(truth_str, best_hyp_str)
                total_error += error
                total_length += tlength
        
        losses.update(raw_loss.mean().item())

    CER = total_error / total_length
    return CER, losses

def make_dataloader(conf, dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=conf.batch_size,
        collate_fn=dataset.pad_collate,
        shuffle=True, num_workers=conf.num_workers
    )
    return dataloader

def train(conf):
    # load model
    if conf.model_name == 'rnnctc':
        model = RNNCTC(conf)
    elif conf.model_name == 'deepspeech2':
        model = DeepSpeech2(conf)
    model = model.cuda()
    conf.logger.info("Model loading complete")

    # dataloader
    TrainDataset = eval(conf.train.dataset)('train', conf)
    DevDataset = eval(conf.dev.dataset)('train', conf)
    train_loader = make_dataloader(conf, TrainDataset)
    dev_loader = make_dataloader(conf, DevDataset)
    conf.logger.info("Data loading complete")

    with open(conf.data.label_dir, 'r') as f:
        label_set = json.load(f)

    optimizer = optim.Adam(model.parameters(), lr=conf.lr)

    conf.logger.info("Training Start")

    min_CER = 1.0
    # training
    for epoch in range(conf.epochs):
        conf.logger.info("-" * 60)
        losses = train_one_epoch(conf, train_loader, model, optimizer, epoch, conf.print_freq, conf.saved_freq, conf.saved_dir, conf.lr)
        conf.logger.info(f"Epoch {epoch} Training done, Loss = {losses.val:.4f} ({losses.avg:.4f})")

        if epoch % conf.saved_freq == 0:
            save_model(conf, model, f"epoch_{epoch}")
            save_model(conf, model, "last")
            conf.logger.info(f"Model saved: epoch_{epoch}.pt, last.pt")

        # validation
        if epoch % conf.val_freq == 0:
            conf.logger.info("Validating...")
            CER, losses = validate(conf, model, epoch, dev_loader, torch.device("cuda"), label_set)
            conf.logger.info(f"Epoch {epoch} Validation done, Loss = {losses.val:.4f} ({losses.avg:.4f}), CER = {100.0*CER:.2f}%")

            # best model save
            if CER < min_CER:
                save_model(conf, model, "best")
                min_CER = CER
                conf.logger.info(f"Model saved: best.pt")
                
                
def evaluate(conf):
    # load model
    if conf.model_name == 'rnnctc':
        model = RNNCTC(conf)
    elif conf.model_name == 'deepspeech2':
        model = DeepSpeech2(conf)
    conf.logger.info(f"Model checkpoint: {conf.checkpoint}")
    model.load_state_dict(torch.load(conf.checkpoint))
    model = model.cuda()
    conf.logger.info("Model loading complete")

    # dataloader
    TestDataset = eval(conf.train.dataset)('test', conf)
    test_loader = make_dataloader(conf, TestDataset)
    conf.logger.info("Data loading complete")

    with open(conf.data.label_dir, 'r') as f:
        label_set = json.load(f)

    conf.logger.info("Evaluation Start")
    CER, losses = validate(conf, model, None, test_loader, torch.device("cuda"), label_set)
    conf.logger.info(f"Evaluation done, Loss = {losses.val:.4f} ({losses.avg:.4f}), CER = {100.0*CER:.2f}%")


def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description='Baseline ASR training')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--model_name', type=str, default='rnnctc', help='ASR model [rnnctc, deepspeech2]')
    parser.add_argument('--config_file', type=str, help='configuration file name')
    parser.add_argument("--saved_dir", type=str, default='runs', help='directory to save logs and models')

    parser.add_argument('--val_freq', type=int, default=2, help='validation frequency')
    parser.add_argument('--epochs', type=int, default=100, help='maximum epochs for training')
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency (in batches)')
    parser.add_argument('--saved_freq', type=int, default=2, help='checkpoint save frequency.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader num_workers')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--beam_width', type=int, default=10, help='CTC beam decoder width')

    parser.add_argument('--checkpoint', type=str, default=None, help='path to saved model')
    parser.add_argument('--eval', action='store_true', help='evaluation flag')

    args = parser.parse_args()

    # merge configurations
    conf_path = os.path.join('./config', args.config_file)
    conf = edict(yaml.load(open(conf_path), Loader=yaml.SafeLoader))
    for k, v in vars(args).items():
        conf[k] = v

    # make directories
    conf.saved_dir = os.path.join(conf.saved_dir, "eval" if conf.eval else "train")
    os.makedirs(conf.saved_dir, exist_ok=True)
    previous_runs = os.listdir(conf.saved_dir) + ['0']
    new_run = str(int(max(previous_runs)) + 1)
    conf.saved_dir = os.path.join(conf.saved_dir, new_run)
    os.makedirs(conf.saved_dir)
    with open(f'{conf.saved_dir}/config.json', 'w') as f:
        json.dump(conf, f, indent=4)

    conf.logger = get_logger(os.path.join(conf.saved_dir, "log.txt"))

    # seed
    conf.logger.info(f'Seed {conf.seed}')
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)

    # Train
    if conf.eval == False:
        train(conf)
        conf.logger.info('-' * 60)
        conf.logger.info('Training complete')
    else:
        evaluate(conf)
        

if __name__ == '__main__':
    main()
