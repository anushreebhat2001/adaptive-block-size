#!/bin/bash
#SBATCH --job-name=adb_smoke_base
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.err

# LLaDA-Base smoke. Tests whether switching from Instruct to Base resolves
# the "model bails after one block, scheduler has no leverage" problem we
# saw in the full eval. Base model:
#   - has no chat template (encode_prompt falls back to plaintext few-shot)
#   - doesn't emit <|eot_id|> by training; continues raw text naturally
#   - paired with min_new_tokens=128 as a backstop
#
# Output goes to ${EAI_SMOKE_DIR}/base/ so the existing Instruct smoke
# results in ${EAI_SMOKE_DIR}/ remain available for comparison.

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

SMOKE_DIR=${EAI_SMOKE_DIR}/base
mkdir -p "${SMOKE_DIR}/labels" "${SMOKE_DIR}/ckpt" "${SMOKE_DIR}/results"

run_in_singularity "
  set -euo pipefail

  echo '[smoke-base] step 1/3: build_teacher_labels (LLaDA-Base, GSM8K, debug first 30 boundaries)'
  python -u -m src.data.build_teacher_labels \
    --model llada --model_id GSAI-ML/LLaDA-8B-Base \
    --benchmark gsm8k \
    --split train --n_prompts 25 \
    --out_dir ${SMOKE_DIR}/labels \
    --shard_size 1000 \
    --max_new_tokens 256 \
    --n_denoise_steps 32 \
    --default_block_size 16 \
    --min_new_tokens 128 \
    --debug_n_boundaries 30 \
    --dtype bf16

  echo '[smoke-base] step 2/3: train predictor (5 epochs)'
  python -u -m src.predictor.train \
    --shard_glob '${SMOKE_DIR}/labels/llada/gsm8k/*.pt' \
    --out_ckpt   ${SMOKE_DIR}/ckpt/llada_teacher.pt \
    --label_source teacher --model llada \
    --hidden_dim 4096 \
    --epochs 5 --batch_size 256 --val_frac 0.2 \
    --class_weight_mode none --label_smoothing 0.1

  echo '[smoke-base] step 3/3: eval ours-teacher vs adablock on 15 GSM8K prompts'
  python -u -m src.eval.run_benchmarks \
    --model llada --model_id GSAI-ML/LLaDA-8B-Base \
    --benchmark gsm8k \
    --split test --n_prompts 15 \
    --max_new_tokens 256 --n_denoise_steps 32 \
    --min_new_tokens 128 \
    --scheduler ours-teacher \
    --predictor ${SMOKE_DIR}/ckpt/llada_teacher.pt \
    --initial_block_size 16 \
    --out ${SMOKE_DIR}/results/ours_teacher.json \
    --dtype bf16

  python -u -m src.eval.run_benchmarks \
    --model llada --model_id GSAI-ML/LLaDA-8B-Base \
    --benchmark gsm8k \
    --split test --n_prompts 15 \
    --max_new_tokens 256 --n_denoise_steps 32 \
    --min_new_tokens 128 \
    --scheduler adablock \
    --initial_block_size 16 \
    --out ${SMOKE_DIR}/results/adablock.json \
    --dtype bf16

  echo '[smoke-base] done.'
  echo '  Pass criteria:'
  echo '    1. Debug-print decoded blocks show real text (not <|endoftext|> padding)'
  echo '    2. boundaries-per-prompt >= 6 (buf count after 25 prompts >= 150)'
  echo '    3. ours-teacher and adablock block_size_histograms differ visibly'
  echo '    4. Eval tokens_per_second differs between the two schedulers'
"
