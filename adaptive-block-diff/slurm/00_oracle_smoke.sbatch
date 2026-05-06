#!/bin/bash
#SBATCH --job-name=adb_oracle_smoke_instruct
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.err

# LLaDA-Instruct oracle smoke. Uses default model_id (LLaDA-8B-Instruct).
# Output goes to ${EAI_SMOKE_DIR}/instruct/oracle/ alongside the Instruct
# teacher smoke.
#
# Run AFTER 00_smoke.sbatch produces healthy teacher labels.
# Prior runs of this configuration showed that even on Instruct's
# 16-token outputs, oracle's argmax(-PPL - lambda/L) does still produce
# 4-class labels because PPL varies prompt-by-prompt across L.

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

SMOKE_DIR=${EAI_SMOKE_DIR}/instruct/oracle
mkdir -p "${SMOKE_DIR}/labels" "${SMOKE_DIR}/ckpt" "${SMOKE_DIR}/results"

run_in_singularity "
  set -euo pipefail

  echo '[oracle-smoke-instruct] step 1/3: build_oracle_labels (LLaDA-Instruct, 12 GSM8K prompts x K=4 rollouts)'
  python -u -m src.data.build_oracle_labels \
    --model llada \
    --benchmark gsm8k \
    --split train --n_prompts 12 \
    --out_dir ${SMOKE_DIR}/labels \
    --shard_size 1000 \
    --max_new_tokens 256 \
    --n_denoise_steps 32 \
    --default_block_size 16 \
    --lam 0.05 \
    --dtype bf16

  echo '[oracle-smoke-instruct] step 2/3: train predictor on oracle labels (5 epochs)'
  python -u -m src.predictor.train \
    --shard_glob '${SMOKE_DIR}/labels/llada/gsm8k/*.pt' \
    --out_ckpt   ${SMOKE_DIR}/ckpt/llada_oracle.pt \
    --label_source oracle --model llada \
    --hidden_dim 4096 \
    --epochs 5 --batch_size 256 --val_frac 0.2 \
    --class_weight_mode none --label_smoothing 0.1

  echo '[oracle-smoke-instruct] step 3/3: eval ours-oracle vs adablock on 15 GSM8K prompts'
  python -u -m src.eval.run_benchmarks \
    --model llada \
    --benchmark gsm8k \
    --split test --n_prompts 15 \
    --max_new_tokens 256 --n_denoise_steps 32 \
    --scheduler ours-oracle \
    --predictor ${SMOKE_DIR}/ckpt/llada_oracle.pt \
    --initial_block_size 16 \
    --out ${SMOKE_DIR}/results/ours_oracle.json \
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

  echo '[oracle-smoke-instruct] done.'
  echo '  Comparison signals (vs Base oracle smoke):'
  echo '    1. Class balance: Instruct usually 4/4 (PPL varies); Base 4/4 better-balanced'
  echo '    2. Val gain over majority: Instruct prior +0.20; Base prior +0.06'
  echo '    3. Eval block_size_histogram: Instruct collapsed (no leverage); Base varied'
"
