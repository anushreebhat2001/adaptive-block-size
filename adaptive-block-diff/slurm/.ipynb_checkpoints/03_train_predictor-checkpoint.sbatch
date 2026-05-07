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
#SBATCH --array=0-3

# Array index -> (model, label_source, hidden_dim).
# 0: llada teacher (LLaDA-Base, hidden 4096)
# 1: llada oracle  (LLaDA-Base, hidden 4096)
# 2: dream teacher (Dream-Base, hidden 3584)
# 3: dream oracle  (Dream-Base, hidden 3584)

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

CONFIGS=(
  "llada teacher 4096 base"
  "llada oracle  4096 lam_15"
  "dream teacher 3584 base"
  "dream oracle  3584 lam_15"
  "llada oracle  4096 lam_10"
  "dream oracle  3584 lam_10"
  "llada oracle  4096 lam_5"
  "dream oracle  3584 lam_5"
)

IFS=' ' read -r MODEL LABEL_SOURCE HIDDEN_DIM TREE <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "${EAI_CKPT_DIR}/base"

run_in_singularity "
  python -u -m src.predictor.train \
    --shard_glob '${EAI_LABEL_DIR}/${TREE}/${LABEL_SOURCE}/${MODEL}/*/*.pt' \
    --out_ckpt ${EAI_CKPT_DIR}/base/${MODEL}_${LABEL_SOURCE}_${TREE}.pt \
    --label_source ${LABEL_SOURCE} \
    --model ${MODEL} \
    --hidden_dim ${HIDDEN_DIM} \
    --epochs 16 \
    --batch_size 512 \
    --lr 3e-4 \
    --weight_decay 5e-2 \
    --val_frac 0.1 \
    --class_weight_mode none \
    --focal_gamma 2.0
"
# weight_decay=5e-2 (up from 1e-2): trajectory showed val_loss climbing past
# epoch ~10 with the old setting, indicating overfit. Stronger L2 should
# flatten that curve so the saved best-val-loss ckpt is closer to the
# best-val-acc point.
