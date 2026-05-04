#!/bin/bash
#SBATCH --job-name=adb_train
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%A_%a.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%A_%a.err
#SBATCH --array=0-1

# Array index -> (model, label_source). LLaDA-only.
# 0: llada teacher
# 1: llada oracle

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

CONFIGS=(
  "llada teacher 4096"
  "llada oracle  4096"
)

IFS=' ' read -r MODEL LABEL_SOURCE HIDDEN_DIM <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

run_in_singularity "
  python -u -m src.predictor.train \
    --shard_glob '${EAI_LABEL_DIR}/${LABEL_SOURCE}/${MODEL}/*/*.pt' \
    --out_ckpt   ${EAI_CKPT_DIR}/${MODEL}_${LABEL_SOURCE}.pt \
    --label_source ${LABEL_SOURCE} \
    --model ${MODEL} \
    --hidden_dim ${HIDDEN_DIM} \
    --epochs 16 \
    --batch_size 512 \
    --lr 3e-4 \
    --weight_decay 1e-2 \
    --val_frac 0.1 \
    --class_weight_mode none \
    --label_smoothing 0.1
"
