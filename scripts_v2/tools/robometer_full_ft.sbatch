#!/bin/bash
# Full fine-tuning of Robometer-4B by dense value distillation, on one H200 (141 GB). --trainable full trains the whole
# language model plus the progress head (4.03B parameters) with the vision encoder frozen, as in Robometer's own full
# fine-tuning. Each step fits Robometer's progress on 8 short clips to a value critic's V(s) (no preference loss).
#
# Usage: fill in the five EDIT values below, then from the folder that contains scripts_v2/:
#   sbatch scripts_v2/tools/robometer_full_ft.sbatch
#
# Memory: fp32 weights, gradients and AdamW moments take about 64 GB, plus about 10 GB of activations.
# Disk: an adapter_step<N>.pt of about 16 GB at every evaluation (every 500 steps, so 8 in all) and a trainer_state.pt
# of about 32 GB. A requeued or resubmitted job continues from the latest adapter_step<N>.pt.
#SBATCH -p EDIT_PARTITION
#SBATCH -A EDIT_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 48:00:00
#SBATCH --requeue
#SBATCH -J rbm_full_distill
#SBATCH -o logs/slurm_rbm_full_distill_%j.out

set -euo pipefail
PY=EDIT_PYTHON          # python of the environment with the robometer package installed
DATA=EDIT_DATA          # the real_to_sim_rewards folder holding the three CubeStack_* datasets
export HF_HOME=EDIT_HF_HOME  # where Robometer-4B (about 9 GB) is downloaded on the first run
export UNSLOTH_DISABLE_STATISTICS=1  # otherwise Unsloth can stall for 120 s on a HuggingFace check and crash

# The .jsonl evaluation files store absolute video paths from the machine that made them; point them at $DATA.
OLD=/scr/shared/real_to_sim_rewards
grep -rl --include='*.jsonl' "$OLD" "$DATA" | xargs -r sed -i "s#$OLD#$DATA#g"

OUT=./logs/robometer_pref/v3_full_distill_lr5e-7
RESUME=""
if ls $OUT/adapter_step*.pt >/dev/null 2>&1; then RESUME="--resume"; fi

$PY -u scripts_v2/tools/finetune_robometer.py --camera front --trainable full \
    --pairs_per_step 0 --distill_clips 8 --progress_reduce cum_delta \
    --train_dataset $DATA/CubeStack_SuccessFailure_v3 --num_train_eval_episodes 100 \
    --val_pairs $DATA/CubeStack_SuccessFailure_val/preference_pairs/random_L30_n1000.jsonl --num_val_pairs 500 \
    --val_dataset $DATA/CubeStack_SuccessFailure_val --num_val_clips 200 \
    --kendall_val_dataset $DATA/CubeStack_KendallTriplets_val \
    --kendall_binary_pairs $DATA/CubeStack_SuccessFailure_val/kendall_binary_pairs.jsonl \
    --lr 5e-7 --warmup_steps 50 --max_steps 4000 --eval_every 500 --save_every 0 \
    --out_dir $OUT --seed 0 $RESUME
