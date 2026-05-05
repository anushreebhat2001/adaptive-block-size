#!/bin/bash
#SBATCH --job-name=adb_oracle
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%A_%a.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%A_%a.err
#SBATCH --array=0-3%4

# Same (model, benchmark) array layout as 01_build_teacher_labels.sbatch.

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

# LLaDA-Base only. Switched from Instruct after Instruct's full pipeline
# showed identical accuracy across all schedulers (no deployment win).
# Base smoke oracle labels: 4/4 classes, H=1.257 (vs Instruct 1.143),
# max class 39.6%. Dream pairs excluded as before.
#
# Output goes to ${EAI_LABEL_DIR}/base/oracle/ so existing Instruct
# labels at ${EAI_LABEL_DIR}/oracle/ remain available for paper baseline.
PAIRS=(
  "llada gsm8k"
  "llada math"
  "llada mbpp"
  "llada owt"
)

IFS=' ' read -r MODEL BENCHMARK <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"

# Oracle is K-rollout (K=4): ~4x cost of teacher. Rely on requeue
# (--signal=B:USR1@120) if a benchmark overruns the 12h budget.
#
# lam=0.05 chosen from the Base lambda sweep on the smoke shards. At this
# value the four classes hit 29/31/29/10% — the most uniform split in the
# sweep, with the lowest ECE (0.036). Predictor val_acc on the smoke is
# noise (val=16 examples), so we picked by class balance + calibration.
# At full pipeline scale the data volume below should resolve the
# predictor accuracy.
#
# n_prompts 2000 (up from 1500): 33% more oracle boundaries (~128k total
# across 4 benchmarks at ~16 boundaries/prompt). Predictor on 4096-dim
# features wants this much data to actually learn structure, not just
# memorize majority class.
run_in_singularity "
  python -u -m src.data.build_oracle_labels \
    --model ${MODEL} --model_id GSAI-ML/LLaDA-8B-Base \
    --benchmark ${BENCHMARK} \
    --split train \
    --n_prompts 2000 \
    --out_dir ${EAI_LABEL_DIR}/base/oracle \
    --shard_size 512 \
    --max_new_tokens 384 \
    --default_block_size 16 \
    --n_denoise_steps 32 \
    --min_new_tokens 128 \
    --lam 0.05 \
    --dtype bf16
"
