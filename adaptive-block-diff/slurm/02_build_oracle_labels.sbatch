#!/bin/bash
#SBATCH --job-name=adb_oracle
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
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

# Oracle is K-rollout (K=4): ~4x cost of teacher. --time=24h with --requeue
# handles overrun.
#
# lam=0.05 chosen from Base lambda sweep on smoke shards: best class
# balance (29/31/29/10 across B={4,8,16,32}) and lowest ECE (0.036) in
# the sweep. Lower lam values (e.g. 0.01) skewed labels toward smaller
# blocks; higher (0.1+) under-represented B=4. 0.05 is the most balanced
# label distribution for the predictor to learn from.
run_in_singularity "
  python -u -m src.data.build_oracle_labels \
    --model ${MODEL} --model_id GSAI-ML/LLaDA-8B-Base \
    --benchmark ${BENCHMARK} \
    --split train \
    --n_prompts 500 \
    --out_dir ${EAI_LABEL_DIR}/base/oracle \
    --shard_size 512 \
    --max_new_tokens 384 \
    --default_block_size 16 \
    --n_denoise_steps 32 \
    --min_new_tokens 128 \
    --lam 0.05 \
    --dtype bf16
"
