#!/bin/bash
#SBATCH --job-name=adb_smoke
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.err

# Verification gate 1 ("Pipeline smoke") from README.md.
# Runs end-to-end on 100 MBPP prompts in ~25 min on a single A100.
# MBPP is used (not GSM8K) because GSM8K outputs are too short on LLaDA
# to produce more than ~1 boundary per prompt; MBPP has multi-paragraph
# code outputs that exercise the scheduler properly.

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

SMOKE_DIR=${EAI_SMOKE_DIR}
mkdir -p "${SMOKE_DIR}/labels" "${SMOKE_DIR}/ckpt" "${SMOKE_DIR}/results"

run_in_singularity "
  set -euo pipefail

  echo '[smoke] step 1/3: build_teacher_labels (MBPP, 256 tokens, debug first 30 boundaries)'
  python -u -m src.data.build_teacher_labels \
    --model llada --benchmark mbpp \
    --split train --n_prompts 100 \
    --out_dir ${SMOKE_DIR}/labels \
    --shard_size 200 \
    --max_new_tokens 256 \
    --n_denoise_steps 32 \
    --default_block_size 16 \
    --debug_n_boundaries 30 \
    --dtype bf16

  echo '[smoke] step 2/3: train predictor (1 epoch)'
  python -u -m src.predictor.train \
    --shard_glob '${SMOKE_DIR}/labels/llada/mbpp/*.pt' \
    --out_ckpt   ${SMOKE_DIR}/ckpt/llada_teacher.pt \
    --label_source teacher --model llada \
    --hidden_dim 4096 \
    --epochs 1 --batch_size 256 --val_frac 0.2

  echo '[smoke] step 3/3: eval ours-teacher vs adablock on 50 MBPP prompts'
  python -u -m src.eval.run_benchmarks \
    --model llada --benchmark mbpp \
    --split test --n_prompts 50 \
    --max_new_tokens 256 --n_denoise_steps 32 \
    --scheduler ours-teacher \
    --predictor ${SMOKE_DIR}/ckpt/llada_teacher.pt \
    --initial_block_size 16 \
    --out ${SMOKE_DIR}/results/ours_teacher.json \
    --dtype bf16

  python -u -m src.eval.run_benchmarks \
    --model llada --benchmark mbpp \
    --split test --n_prompts 50 \
    --max_new_tokens 256 --n_denoise_steps 32 \
    --scheduler adablock \
    --initial_block_size 16 \
    --out ${SMOKE_DIR}/results/adablock.json \
    --dtype bf16

  echo '[smoke] done. Compare block-size histograms in:'
  echo '  ${SMOKE_DIR}/results/ours_teacher.json'
  echo '  ${SMOKE_DIR}/results/adablock.json'
"
