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
# the inner training loop fast (10 epochs x ~10 lambdas x small dataset).
#
# Run this AFTER 02_build_oracle_labels.sbatch finishes.

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

run_in_singularity "
  python -u -m scripts.lambda_sweep \
    --shard_glob '${EAI_LABEL_DIR}/oracle/llada/*/*.pt' \
    --hidden_dim 4096 \
    --lambdas 0.0,0.005,0.01,0.05,0.1,0.5,1.0 \
    --epochs 10 \
    --batch_size 256 \
    --val_frac 0.1 \
    --out ${EAI_RESULT_DIR}/lambda_sweep_llada_oracle.json
"
