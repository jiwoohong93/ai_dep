#!/bin/sh
# Train + evaluate one model with the final setting (DINOv2 ViT-B/14, optic disc crop, 448 px).
# usage: sh fairfundus_train_eval.sh <erm|fair> <gpu_id> [seed] [extra main.py args]
#   erm  : baseline (before the fairness method)
#   fair : FairFundusNet (fair batch + group loss reweighting)
# evaluate released weights: sh fairfundus_eval.sh <erm|fair> <gpu_id>
METHOD=${1:-fair}; GPU=${2:-0}; SEED=${3:-0}
[ $# -ge 3 ] && shift 3 || shift $#
CUDA_VISIBLE_DEVICES=$GPU python main.py --method $METHOD --random_seed $SEED \
    --model dinov2_b --view disc --img_size 448 --augment --amp \
    --optimizer adamw --blr 1e-5 --head_lr_scale 100 --weight_decay 0.05 \
    --batch_size 32 --total_epochs 60 --warmup_epochs 5 "$@"
