#!/bin/bash
#SBATCH --job-name=adb_smoke_instruct
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.err

# LLaDA-Instruct smoke. Uses default model_id (GSAI-ML/LLaDA-8B-Instruct).
# Output goes to ${EAI_SMOKE_DIR}/instruct/ so it sits alongside the Base
# and Dream smokes for direct comparison.
#
# Note: prior runs of this exact configuration showed Instruct teacher
# labels collapse to 2 classes (B=4 ~1%, B=32 ~99%) because Instruct bails
# after one block of EOS tokens. We re-run anyway to verify (a) the new
# HPC env loads the model, (b) the runner produces valid shards, (c)
# whether anything has changed.

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

SMOKE_DIR=${EAI_SMOKE_DIR}/instruct
mkdir -p "${SMOKE_DIR}/labels" "${SMOKE_DIR}/ckpt" "${SMOKE_DIR}/results"

run_in_singularity "
  set -euo pipefail

  echo '[smoke-instruct] step 1/3: build_teacher_labels (LLaDA-Instruct, GSM8K, debug first 30 boundaries)'
  python -u -m src.data.build_teacher_labels \
    --model llada \
    --benchmark gsm8k \
    --split train --n_prompts 25 \
    --out_dir ${SMOKE_DIR}/labels \
    --shard_size 1000 \
    --max_new_tokens 256 \
    --n_denoise_steps 32 \
    --default_block_size 16 \
    --debug_n_boundaries 30 \
    --dtype bf16

  echo '[smoke-instruct] step 2/3: train predictor (5 epochs)'
  python -u -m src.predictor.train \
    --shard_glob '${SMOKE_DIR}/labels/llada/gsm8k/*.pt' \
    --out_ckpt   ${SMOKE_DIR}/ckpt/llada_teacher.pt \
    --label_source teacher --model llada \
    --hidden_dim 4096 \
    --epochs 5 --batch_size 256 --val_frac 0.2 \
    --class_weight_mode none --label_smoothing 0.1

  echo '[smoke-instruct] step 3/3: eval ours-teacher vs adablock on 15 GSM8K prompts'
  python -u -m src.eval.run_benchmarks \
    --model llada \
    --benchmark gsm8k \
    --split test --n_prompts 15 \
    --max_new_tokens 256 --n_denoise_steps 32 \
    --scheduler ours-teacher \
    --predictor ${SMOKE_DIR}/ckpt/llada_teacher.pt \
    --initial_block_size 16 \
    --out ${SMOKE_DIR}/results/ours_teacher.json \
    --dtype bf16

  python -u -m src.eval.run_benchmarks \
    --model llada \
    --benchmark gsm8k \
    --split test --n_prompts 15 \
    --max_new_tokens 256 --n_denoise_steps 32 \
    --scheduler adablock \
    --initial_block_size 16 \
    --out ${SMOKE_DIR}/results/adablock.json \
    --dtype bf16

  echo '[smoke-instruct] done.'
  echo '  Comparison signals (vs Base smoke at same config):'
  echo '    1. Decoded blocks: Instruct shows mostly <|endoftext|>; Base shows real reasoning'
  echo '    2. Boundaries-per-prompt: Instruct ~1; Base ~16'
  echo '    3. Class distribution: Instruct collapses; Base 4/4'
"
