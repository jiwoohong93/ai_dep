CUDA_VISIBLE_DEVICES=0 python main.py --task cls --usage lp --dataset PAPILA --model BiomedCLIP \
    --sensitive_name Age --method erm --total_epochs 300 --warmup_epochs 5 --blr 5e-4 \
    --batch_size 128 --optimizer adamw --min_lr 1e-10 --weight_decay 0.05 --lr_decay_rate 0.7 --random_seed 2

# CUDA_VISIBLE_DEVICES=0 python main.py --task cls --usage lp --dataset PAPILA --model BiomedCLIP \
#     --sensitive_name Age --method erm --total_epochs 50 --warmup_epochs 5 --blr 5e-4 \
#     --batch_size 128 --optimizer adamw --min_lr 1e-4 --weight_decay 0.05 --lr_decay_rate 0.7 --random_seed 2
