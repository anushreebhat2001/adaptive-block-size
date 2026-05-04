#!/bin/bash
#SBATCH --job-name=adb_smoke
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.err

# Verification gate 1 ("Pipeline smoke") from README.md.
# Runs end-to-end on GSM8K in ~30 min on a single A100.
#
# Uses LLaDA-8B (BASE, not Instruct) because the chat-tuned Instruct model
# bails out after one block on most short-answer problems, leaving only ~1
# boundary per prompt and rendering the scheduler invisible. Base
# continues raw text naturally and produces multi-paragraph CoT from the
# few-shot plaintext prompt, giving 8-18 boundaries per prompt.
#
# min_new_tokens=128 is a backstop in case the model still wants to emit
# EOS-like tokens early -- it forces at least 8 blocks of 16 tokens.

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

SMOKE_DIR=${EAI_SMOKE_DIR}
mkdir -p "${SMOKE_DIR}/labels" "${SMOKE_DIR}/ckpt" "${SMOKE_DIR}/results"

run_in_singularity "
  set -euo pipefail

  echo '[smoke] step 1/3: build_teacher_labels (LLaDA-Base, GSM8K, 384 tokens, debug first 30 boundaries)'
  python -u -m src.data.build_teacher_labels \
    --model llada --model_id GSAI-ML/LLaDA-8B \
    --benchmark gsm8k \
    --split train --n_prompts 100 \
    --out_dir ${SMOKE_DIR}/labels \
    --shard_size 1000 \
    --max_new_tokens 384 \
    --n_denoise_steps 32 \
    --default_block_size 16 \
    --min_new_tokens 128 \
    --debug_n_boundaries 30 \
    --dtype bf16

  echo '[smoke] step 2/3: train predictor (5 epochs, report majority baseline)'
  python -u -m src.predictor.train \
    --shard_glob '${SMOKE_DIR}/labels/llada/gsm8k/*.pt' \
    --out_ckpt   ${SMOKE_DIR}/ckpt/llada_teacher.pt \
    --label_source teacher --model llada \
    --hidden_dim 4096 \
    --epochs 5 --batch_size 256 --val_frac 0.2

  echo '[smoke] step 3/3: eval ours-teacher vs adablock on 50 GSM8K prompts'
  python -u -m src.eval.run_benchmarks \
    --model llada --model_id GSAI-ML/LLaDA-8B \
    --benchmark gsm8k \
    --split test --n_prompts 50 \
    --max_new_tokens 384 --n_denoise_steps 32 \
    --min_new_tokens 128 \
    --scheduler ours-teacher \
    --predictor ${SMOKE_DIR}/ckpt/llada_teacher.pt \
    --initial_block_size 16 \
    --out ${SMOKE_DIR}/results/ours_teacher.json \
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

  echo '[smoke] done. Compare block-size histograms in:'
  echo '  ${SMOKE_DIR}/results/ours_teacher.json'
  echo '  ${SMOKE_DIR}/results/adablock.json'
"
