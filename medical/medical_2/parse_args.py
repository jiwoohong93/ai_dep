import argparse


def collect_args():
    parser = argparse.ArgumentParser(description="FairFundusNet: glaucoma classification on PAPILA")

    # experiment
    parser.add_argument("--method", default="fair", choices=["erm", "fair", "resample"],
                        help="erm: baseline (shuffled batches, class-weighted cross entropy). "
                             "fair: fair batch sampling + group loss reweighting. "
                             "resample: every batch holds the same number of images from each "
                             "(age group x diagnosis) cell, plain cross entropy.")
    parser.add_argument("--model", default="biomedclip",
                        choices=["biomedclip", "dinov2_b", "dinov2_l"])
    parser.add_argument("--view", default="full", choices=["full", "disc"],
                        help="full: whole image. disc: fixed central square around the optic disc "
                             "(0.6 x image height; PAPILA images are disc-centred).")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--augment", action="store_true",
                        help="training augmentation: random resized crop, flip, small rotation, colour jitter")
    parser.add_argument("--dataset", default="PAPILA", choices=["PAPILA"])
    parser.add_argument("--sensitive_name", default="Age", choices=["Age", "Sex"])
    parser.add_argument("--age_threshold", type=int, default=69,
                        help="sensitive attribute Age: group 1 if age >= threshold, else group 0")
    parser.add_argument("--fair_batch_ratio", type=float, default=0.5,
                        help="fair only: share of group 1 in every batch")
    parser.add_argument("--group_loss_weights", type=float, nargs=2, default=None, metavar=("W0", "W1"),
                        help="fair only: loss weights of group 0 and group 1 "
                             "(default: the other group's share of the training set)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="name of the output sub-folder (default: the method name)")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--exp_path", type=str, default="./output")

    # evaluation only
    parser.add_argument("--eval_only", action="store_true",
                        help="skip training; load --ckpt_path and evaluate on the test set")
    parser.add_argument("--ckpt_path", type=str, default="",
                        help="checkpoint for --eval_only (default: <save_folder>/ckpt_best.pth)")

    # training
    parser.add_argument("--total_epochs", type=int, default=300)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--blr", type=float, default=5e-4, help="base learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-10)
    parser.add_argument("--head_lr_scale", type=float, default=1.0,
                        help="learning rate of the classification head relative to the encoder")
    parser.add_argument("--amp", action="store_true", help="mixed precision training")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--eval_freq", type=int, default=1,
                        help="evaluate on the test set every N epochs")
    parser.add_argument("--target_acc", type=float, default=0.94,
                        help="fair only: checkpoints reaching this accuracy and --target_gap are preferred as best")
    parser.add_argument("--target_gap", type=float, default=0.02)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--no_cls_balance", dest="cls_balance", action="store_false",
                        help="disable class (diagnosis) weights in the cross entropy")
    parser.add_argument("--no_cuda", dest="cuda", action="store_false")

    return parser.parse_args()
