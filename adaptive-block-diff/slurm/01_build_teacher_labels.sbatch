#!/bin/bash
#SBATCH --job-name=adb_teacher
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%A_%a.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%A_%a.err
#SBATCH --array=0-3%4

# Job array index -> (model, benchmark) pair.
# 0..3: llada x {gsm8k, math, mbpp, owt}
# 4..7: dream x {gsm8k, math, mbpp, owt}

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

# LLaDA-Instruct on (gsm8k, math, mbpp, owt). Validated configuration:
# 4-class oracle labels, gain_over_majority=+0.22 on the smoke. Dream-
# Instruct is empirically worse (oracle labels collapse to 2 classes),
# so we don't include Dream pairs.
PAIRS=(
  "llada gsm8k"
  "llada math"
  "llada mbpp"
  "llada owt"
)

IFS=' ' read -r MODEL BENCHMARK <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"

run_in_singularity "
  python -u -m src.data.build_teacher_labels \
    --model ${MODEL} \
    --benchmark ${BENCHMARK} \
    --split train \
    --n_prompts 5000 \
    --out_dir ${EAI_LABEL_DIR}/teacher \
    --shard_size 512 \
    --max_new_tokens 384 \
    --default_block_size 16 \
    --n_denoise_steps 32 \
    --dtype bf16
"
