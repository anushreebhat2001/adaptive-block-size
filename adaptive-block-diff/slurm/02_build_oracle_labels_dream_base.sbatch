#!/bin/bash
#SBATCH --job-name=adb_oracle_dream
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%A_%a.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%A_%a.err
#SBATCH --array=0-3%4

# Dream-Base oracle labels on (gsm8k, math, mbpp, owt). Mirrors
# 02_build_oracle_labels.sbatch but with --model dream and Dream-Base
# checkpoint. K=4 rollouts at lam=0.05 (matches LLaDA-Base oracle).
#
# Dream-Base oracle labels were healthy on the smoke (4/4 classes,
# H=1.26, max class 48%) -- same quality range as LLaDA-Base oracle.
# This run produces the apples-to-apples cross-architecture comparison.

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

PAIRS=(
  "dream gsm8k"
  "dream math"
  "dream mbpp"
  "dream owt"
)

IFS=' ' read -r MODEL BENCHMARK <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"

run_in_singularity "
  python -u -m src.data.build_oracle_labels \
    --model ${MODEL} --model_id Dream-org/Dream-v0-Base-7B \
    --benchmark ${BENCHMARK} \
    --split train \
    --n_prompts 500 \
    --out_dir ${EAI_LABEL_DIR}/base/oracle \
    --shard_size 512 \
    --max_new_tokens 384 \
    --default_block_size 16 \
    --n_denoise_steps 32 \
    --min_new_tokens 128 \
    --lam 0.05 \
    --dtype bf16
"
