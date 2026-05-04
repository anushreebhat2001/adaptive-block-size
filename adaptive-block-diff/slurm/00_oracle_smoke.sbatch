#!/bin/bash
#SBATCH --job-name=adb_oracle_smoke
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.err

# Project go/no-go gate. Runs K=4 oracle rollouts on a small GSM8K slice
# and checks whether oracle labels span >=3 of 4 classes AND a trained
# predictor beats the majority-class baseline.
#
# If this passes, the project's central premise (B genuinely matters and
# is learnable from features) is supported and we scale up.
# If it fails, no amount of teacher-rule tuning will save the predictor;
# stop and pivot.

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

SMOKE_DIR=${EAI_SMOKE_DIR}/oracle
mkdir -p "${SMOKE_DIR}/labels" "${SMOKE_DIR}/ckpt" "${SMOKE_DIR}/results"

run_in_singularity "
  set -euo pipefail

  echo '[oracle-smoke] step 1/3: build_oracle_labels (LLaDA-Instruct, 50 GSM8K prompts x K=4 rollouts)'
  python -u -m src.data.build_oracle_labels \
    --model llada \
    --benchmark gsm8k \
    --split train --n_prompts 50 \
    --out_dir ${SMOKE_DIR}/labels \
    --shard_size 1000 \
    --max_new_tokens 384 \
    --n_denoise_steps 32 \
    --default_block_size 16 \
    --lam 0.05 \
    --dtype bf16

  echo '[oracle-smoke] step 2/3: train predictor on oracle labels (5 epochs)'
  python -u -m src.predictor.train \
    --shard_glob '${SMOKE_DIR}/labels/llada/gsm8k/*.pt' \
    --out_ckpt   ${SMOKE_DIR}/ckpt/llada_oracle.pt \
    --label_source oracle --model llada \
    --hidden_dim 4096 \
    --epochs 5 --batch_size 256 --val_frac 0.2

  echo '[oracle-smoke] step 3/3: eval ours-oracle vs adablock on 50 GSM8K prompts'
  python -u -m src.eval.run_benchmarks \
    --model llada \
    --benchmark gsm8k \
    --split test --n_prompts 50 \
    --max_new_tokens 384 --n_denoise_steps 32 \
    --scheduler ours-oracle \
    --predictor ${SMOKE_DIR}/ckpt/llada_oracle.pt \
    --initial_block_size 16 \
    --out ${SMOKE_DIR}/results/ours_oracle.json \
    --dtype bf16

  python -u -m src.eval.run_benchmarks \
    --model llada \
    --benchmark gsm8k \
    --split test --n_prompts 50 \
    --max_new_tokens 384 --n_denoise_steps 32 \
    --scheduler adablock \
    --initial_block_size 16 \
    --out ${SMOKE_DIR}/results/adablock.json \
    --dtype bf16

  echo '[oracle-smoke] done.'
  echo '  Pass: oracle label_dist spans >=3 classes AND val_acc > val_majority_acc.'
  echo '  Results in: ${SMOKE_DIR}/results/'
"
