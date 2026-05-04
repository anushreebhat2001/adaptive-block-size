#!/bin/bash
#SBATCH --job-name=adb_oracle_smoke_base
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.err

# LLaDA-Base oracle smoke. The real go/no-go for "should we redo the full
# pipeline on Base?" Tests whether oracle labels span 4 classes AND eval
# shows different ours-oracle vs adablock numbers when the model produces
# longer outputs.
#
# Run AFTER 00_smoke_base.sbatch validates that Base produces real text.

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

SMOKE_DIR=${EAI_SMOKE_DIR}/base/oracle
mkdir -p "${SMOKE_DIR}/labels" "${SMOKE_DIR}/ckpt" "${SMOKE_DIR}/results"

run_in_singularity "
  set -euo pipefail

  echo '[oracle-smoke-base] step 1/3: build_oracle_labels (LLaDA-Base, 50 GSM8K prompts x K=4 rollouts)'
  python -u -m src.data.build_oracle_labels \
    --model llada --model_id GSAI-ML/LLaDA-8B \
    --benchmark gsm8k \
    --split train --n_prompts 50 \
    --out_dir ${SMOKE_DIR}/labels \
    --shard_size 1000 \
    --max_new_tokens 384 \
    --n_denoise_steps 32 \
    --default_block_size 16 \
    --min_new_tokens 128 \
    --lam 0.01 \
    --dtype bf16

  echo '[oracle-smoke-base] step 2/3: train predictor on oracle labels (5 epochs)'
  python -u -m src.predictor.train \
    --shard_glob '${SMOKE_DIR}/labels/llada/gsm8k/*.pt' \
    --out_ckpt   ${SMOKE_DIR}/ckpt/llada_oracle.pt \
    --label_source oracle --model llada \
    --hidden_dim 4096 \
    --epochs 5 --batch_size 256 --val_frac 0.2 \
    --class_weight_mode none --label_smoothing 0.1

  echo '[oracle-smoke-base] step 3/3: eval ours-oracle vs adablock on 50 GSM8K prompts'
  python -u -m src.eval.run_benchmarks \
    --model llada --model_id GSAI-ML/LLaDA-8B \
    --benchmark gsm8k \
    --split test --n_prompts 50 \
    --max_new_tokens 384 --n_denoise_steps 32 \
    --min_new_tokens 128 \
    --scheduler ours-oracle \
    --predictor ${SMOKE_DIR}/ckpt/llada_oracle.pt \
    --initial_block_size 16 \
    --out ${SMOKE_DIR}/results/ours_oracle.json \
    --dtype bf16

  python -u -m src.eval.run_benchmarks \
    --model llada --model_id GSAI-ML/LLaDA-8B \
    --benchmark gsm8k \
    --split test --n_prompts 50 \
    --max_new_tokens 384 --n_denoise_steps 32 \
    --min_new_tokens 128 \
    --scheduler adablock \
    --initial_block_size 16 \
    --out ${SMOKE_DIR}/results/adablock.json \
    --dtype bf16

  echo '[oracle-smoke-base] done.'
  echo '  GO criteria for redoing the full pipeline on Base:'
  echo '    1. Oracle label class_balance covers >=3 classes, no class >70%'
  echo '    2. Predictor val gain_over_majority > 0 (anything positive)'
  echo '    3. Eval ours-oracle tokens_per_second OR accuracy differs from adablock'
  echo '         by a margin big enough to publish (>=2 pts acc or >=2 t/s)'
  echo '  NO-GO criteria (stick with Instruct results, write paper as-is):'
  echo '    1. Oracle labels collapse to 1-2 classes'
  echo '    2. Eval still shows identical accuracy across schedulers'
"
