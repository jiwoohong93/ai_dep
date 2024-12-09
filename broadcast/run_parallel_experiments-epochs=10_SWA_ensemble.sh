#!/bin/bash

# List of models to experiment with
models=(
    "klue/roberta-large"
    "klue/roberta-base"
    # "klue/roberta-small"
    # "klue/bert-base"
)

# Configuration parameters
TASK="ner"
EPOCHS=10

outside_weights=(1.0)
batch_sizes=(32)
learning_rates=(5e-5)
swa_lrs=(5e-5)
devices=(
    "cuda:0" "cuda:1" "cuda:2" "cuda:3"
    "cuda:4" "cuda:5" "cuda:6" "cuda:7"
    )
seeds=(1 2 3 4 5 6 7 8)

# Directory for logs
mkdir -p nohup_logs

# Function to run an experiment
run_experiment() {
    local model=$1
    local outside_weight=$2
    local batch_size=$3
    local lr=$4
    local swa_lr=$5
    local seed=$6
    local device=$7
    local experiment_name="${model##*/}_ow${outside_weight}_bs${batch_size}_lr${lr}_seed${seed}_ep${EPOCHS}_SWAlr_${swa_lr}_epochwise"
    local log_file="nohup_logs/${experiment_name}.log"
    
    echo "Running experiment: $experiment_name on $device"
    nohup python main_SWA_NER.py \
        --model "$model" \
        --batch_size $batch_size \
        --epochs $EPOCHS \
        --lr $lr \
        --seed $seed \
        --device $device \
        --outside_weight $outside_weight \
        --swa_start 1 \
        --swa_lr $swa_lr \
        > "$log_file" 2>&1 &
    
    echo $! >> experiment_pids.txt
}

# Clear previous PIDs fiㄴle
> experiment_pids.txt

# Main execution
echo "Starting NER experiments"
echo "----------------------------------------"

experiment_count=0
max_parallel=$((${#devices[@]} - 1))  # 0-based index

for model in "${models[@]}"; do
    for ow in "${outside_weights[@]}"; do
        for bs in "${batch_sizes[@]}"; do
            for lr in "${learning_rates[@]}"; do
                for swa_lr in "${swa_lrs[@]}"; do
                    for seed in "${seeds[@]}"; do
                        device_index=$((experiment_count % ${#devices[@]}))
                        device=${devices[$device_index]}
                        
                        run_experiment "$model" "$ow" "$bs" "$lr" "$swa_lr" "$seed" "$device"
                        
                        experiment_count=$((experiment_count + 1))
                        
                        # Wait if we've reached the maximum number of parallel experiments
                        if [ $((experiment_count % (max_parallel + 1))) -eq 0 ]; then
                            wait
                        fi
                    done
                done
            done
        done
    done
done

# Wait for all experiments to complete
wait

echo "All experiments completed"
echo "To terminate all experiments, run: kill \$(cat experiment_pids_.txt)"
