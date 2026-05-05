#!/bin/bash
#SBATCH --job-name=adb_lambda_sweep
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.err

# Offline lambda sweep on cached oracle shards. Re-derives labels at multiple
# lambda values from the per-L PPLs cached in each shard, retrains the
# predictor on each, and reports val gain + class balance + ECE per lambda.
# CPU-bound after data load -- a GPU is requested only because A100 makes
# the inner training loop fast.
#
# Run this AFTER 02_build_oracle_labels.sbatch finishes.
#
# Sweep grid: dense in [0, 0.1] where the first sweep showed all the
# inflection (gain drops from +0.227 -> +0.136 across this range), sparse
# in [0.2, 1.0] where the curve is monotonic and uninteresting.
# 13 points total -- same wall-clock budget as a uniform 0.05 step but
# much better resolution where it matters.

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

mkdir -p "${EAI_RESULT_DIR}/base"

run_in_singularity "
  python -u -m scripts.lambda_sweep \
    --shard_glob '${EAI_LABEL_DIR}/base/oracle/llada/*/*.pt' \
    --hidden_dim 4096 \
    --lambdas 0.0,0.005,0.01,0.015,0.02,0.03,0.04,0.05,0.075,0.1,0.2,0.5,1.0 \
    --epochs 10 \
    --batch_size 256 \
    --val_frac 0.1 \
    --out ${EAI_RESULT_DIR}/base/lambda_sweep_llada_oracle.json
"
