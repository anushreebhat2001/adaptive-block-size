#!/bin/bash
#SBATCH --job-name=adb_oracle
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%A_%a.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%A_%a.err
#SBATCH --array=0-7%4

# Same (model, benchmark) array layout as 01_build_teacher_labels.sbatch.

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

PAIRS=(
  "llada gsm8k"
  "llada math"
  "llada mbpp"
  "llada owt"
  "dream gsm8k"
  "dream math"
  "dream mbpp"
  "dream owt"
)

IFS=' ' read -r MODEL BENCHMARK <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"

# Oracle is K-rollout (K=4): ~4x cost of teacher. Reduce n_prompts to keep
# the per-job runtime under 8h and rely on requeue if it overruns.
run_in_singularity "
  python -u -m src.data.build_oracle_labels \
    --model ${MODEL} \
    --benchmark ${BENCHMARK} \
    --split train \
    --n_prompts 3500 \
    --out_dir ${EAI_LABEL_DIR}/oracle \
    --shard_size 512 \
    --max_new_tokens 256 \
    --default_block_size 16 \
    --n_denoise_steps 32 \
    --lam 0.05 \
    --dtype bf16
"
