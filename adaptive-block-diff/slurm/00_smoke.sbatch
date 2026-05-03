#!/bin/bash
#SBATCH --job-name=adb_smoke
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.err

# Verification gate 1 ("Pipeline smoke") from README.md.
# Runs end-to-end on 100 GSM8K prompts in ~25 min on a single A100.

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

SMOKE_DIR=${EAI_SMOKE_DIR}
mkdir -p "${SMOKE_DIR}/labels" "${SMOKE_DIR}/ckpt" "${SMOKE_DIR}/results"

run_in_singularity "
  set -euo pipefail

  echo '[smoke] step 1/3: build_teacher_labels'
  python -u -m src.data.build_teacher_labels \
    --model llada --benchmark gsm8k \
    --split train --n_prompts 100 \
    --out_dir ${SMOKE_DIR}/labels \
    --shard_size 100 \
    --max_new_tokens 128 \
    --n_denoise_steps 16 \
    --default_block_size 16 \
    --dtype bf16

  echo '[smoke] step 2/3: train predictor (1 epoch)'
  python -u -m src.predictor.train \
    --shard_glob '${SMOKE_DIR}/labels/llada/gsm8k/*.pt' \
    --out_ckpt   ${SMOKE_DIR}/ckpt/llada_teacher.pt \
    --label_source teacher --model llada \
    --hidden_dim 4096 \
    --epochs 1 --batch_size 256 --val_frac 0.2

  echo '[smoke] step 3/3: eval ours-teacher vs adablock on 50 prompts'
  python -u -m src.eval.run_benchmarks \
    --model llada --benchmark gsm8k \
    --split test --n_prompts 50 \
    --max_new_tokens 128 --n_denoise_steps 16 \
    --scheduler ours-teacher \
    --predictor ${SMOKE_DIR}/ckpt/llada_teacher.pt \
    --initial_block_size 16 \
    --out ${SMOKE_DIR}/results/ours_teacher.json \
    --dtype bf16

  python -u -m src.eval.run_benchmarks \
    --model llada --benchmark gsm8k \
    --split test --n_prompts 50 \
    --max_new_tokens 128 --n_denoise_steps 16 \
    --scheduler adablock \
    --initial_block_size 16 \
    --out ${SMOKE_DIR}/results/adablock.json \
    --dtype bf16

  echo '[smoke] done. Compare block-size histograms in:'
  echo '  ${SMOKE_DIR}/results/ours_teacher.json'
  echo '  ${SMOKE_DIR}/results/adablock.json'
"
