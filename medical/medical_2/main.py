import json
import logging
import os
import random
import time

import numpy as np
import torch

import parse_args
from datasets.utils import get_dataset
from models import build_encoder
from trainers import CLSTrainer
from wrappers import LinearProbeWrapper


def setup_experiment(args):
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.lr = args.blr
    args.run_name = args.run_name or args.method
    args.save_folder = os.path.join(
        args.exp_path, f"age{args.age_threshold}", args.run_name, f"seed{args.random_seed}")
    os.makedirs(args.save_folder, exist_ok=True)

    with open(f"configs/datasets/{args.dataset}.json", "r") as f:
        args.data_setting = json.load(f)
    return args


def setup_logger(folder):
    logger = logging.getLogger("fairfundusnet")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S")
    log_file = os.path.join(folder, "history_{}.log".format(time.strftime("%Y%m%d-%H%M%S")))
    for handler in (logging.FileHandler(log_file, mode="w"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def set_seed(seed, cuda):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = setup_experiment(parse_args.collect_args())
    logger = setup_logger(args.save_folder)
    logger.info(args)
    set_seed(args.random_seed, args.cuda)

    encoder = build_encoder(args.model)
    args.norm_mean, args.norm_std = encoder.mean, encoder.std

    train_data, train_dataloader = get_dataset(args, split="train")
    test_data, test_dataloader = get_dataset(args, split="test")

    model = LinearProbeWrapper(encoder).to(args.device)
    trainer = CLSTrainer(args, model, train_data, test_dataloader, logger)

    if args.eval_only:
        ckpt_path = args.ckpt_path or os.path.join(args.save_folder, "ckpt_best.pth")
        logger.info(f"Eval only, loading {ckpt_path}")
        trainer.load_checkpoint(ckpt_path)
        s = trainer.evaluate(test_dataloader, save_path=os.path.join(args.save_folder, "eval"))
        print("accuracy {:.2f}% | age < {t}: {:.2f}% | age >= {t}: {:.2f}% | accuracy gap {:.2f}%".format(
            100 * s["acc@best_f1"], *(100 * v for v in s["group_acc@best_f1"]), 100 * s["acc_gap@best_f1"],
            t=args.age_threshold))
    else:
        trainer.init_optimizers()
        trainer.train(train_dataloader)
        trainer.evaluate(test_dataloader, save_path=os.path.join(args.save_folder, "final"))
