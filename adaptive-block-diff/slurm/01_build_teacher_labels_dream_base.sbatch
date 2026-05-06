#!/bin/bash
#SBATCH --job-name=adb_teacher_dream
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%A_%a.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%A_%a.err
#SBATCH --array=0-3%4

# Dream-Base teacher labels on (gsm8k, math, mbpp, owt). Mirrors
# 01_build_teacher_labels.sbatch but with --model dream and Dream-Base
# checkpoint. Dream's hidden_dim is 3584 (vs LLaDA 4096) -- the predictor
# in 03 picks this up via the hidden_dim row in CONFIGS there.
#
# Note: Dream-Base teacher labels collapsed to 2/4 classes on the smoke
# (B=4 + B=32 only, no B=8 or B=16). Running the full pipeline anyway to
# (a) confirm the collapse holds at scale, (b) give the predictor a
# chance to learn the binary teacher signal, (c) provide an apples-to-
# apples cross-architecture comparison row for the paper.
#
# Output goes to ${EAI_LABEL_DIR}/base/teacher/dream/ so it sits next to
# llada/ and 03's --shard_glob pattern picks it up automatically.

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

PAIRS=(
  "dream gsm8k"
  "dream math"
  "dream mbpp"
  "dream owt"
)

IFS=' ' read -r MODEL BENCHMARK <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"

run_in_singularity "
  python -u -m src.data.build_teacher_labels \
    --model ${MODEL} --model_id Dream-org/Dream-v0-Base-7B \
    --benchmark ${BENCHMARK} \
    --split train \
    --n_prompts 500 \
    --out_dir ${EAI_LABEL_DIR}/base/teacher \
    --shard_size 512 \
    --max_new_tokens 384 \
    --default_block_size 16 \
    --n_denoise_steps 32 \
    --min_new_tokens 128 \
    --dtype bf16
"
