#!/bin/sh
# Evaluate the released weights in ./release/ on the test set (run download_weights.sh first).
# usage: sh fairfundus_eval.sh <erm|fair> <gpu_id>
#   erm  : release/baseline_vitb.pth   (before the fairness method)
#   fair : release/fairfundusnet.pth   (FairFundusNet)
METHOD=${1:-fair}; GPU=${2:-0}
[ "$METHOD" = "erm" ] && CKPT=release/baseline_vitb.pth || CKPT=release/fairfundusnet.pth
CUDA_VISIBLE_DEVICES=$GPU python main.py --method $METHOD --random_seed 0 --run_name release_$METHOD \
    --model dinov2_b --view disc --img_size 448 --amp --eval_only --ckpt_path $CKPT
