#!/bin/bash
#SBATCH --job-name=adb_teacher
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%A_%a.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%A_%a.err
#SBATCH --array=0-7%4

# Job array index -> (model, benchmark) pair.
# 0..3: llada x {gsm8k, math, mbpp, owt}
# 4..7: dream x {gsm8k, math, mbpp, owt}

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

# (model, model_id, benchmark). LLaDA uses Base (no chat template, naturally
# long generation); Dream stays Instruct (no Base release). min_new_tokens
# is a backstop for the chat-tuned Dream path.
PAIRS=(
  "llada GSAI-ML/LLaDA-8B gsm8k"
  "llada GSAI-ML/LLaDA-8B math"
  "llada GSAI-ML/LLaDA-8B mbpp"
  "llada GSAI-ML/LLaDA-8B owt"
  "dream Dream-org/Dream-v0-Instruct-7B gsm8k"
  "dream Dream-org/Dream-v0-Instruct-7B math"
  "dream Dream-org/Dream-v0-Instruct-7B mbpp"
  "dream Dream-org/Dream-v0-Instruct-7B owt"
)

IFS=' ' read -r MODEL MODEL_ID BENCHMARK <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"

run_in_singularity "
  python -u -m src.data.build_teacher_labels \
    --model ${MODEL} \
    --model_id ${MODEL_ID} \
    --benchmark ${BENCHMARK} \
    --split train \
    --n_prompts 5000 \
    --out_dir ${EAI_LABEL_DIR}/teacher \
    --shard_size 512 \
    --max_new_tokens 384 \
    --default_block_size 16 \
    --n_denoise_steps 32 \
    --min_new_tokens 128 \
    --dtype bf16
"
