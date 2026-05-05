#!/bin/bash
#SBATCH --job-name=adb_eval
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%A_%a.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%A_%a.err
#SBATCH --array=0-27%6

# LLaDA-Base eval. Reads predictor ckpts from ${EAI_CKPT_DIR}/base/ and
# writes results to ${EAI_RESULT_DIR}/base/ so Instruct results from the
# prior pipeline run remain on disk for comparison.
#
# Dropped ifeval: scoring.ifeval_correct returns len(generated)>0, which
# is always 1.0 and contributed the suspicious "ifeval=1.000 across all
# schedulers" line to the prior eval (no real signal).
#
# Array index decomposes into:
#   benchmark  ∈ {gsm8k, math, humaneval, mbpp}            -> 4
#   scheduler  ∈ {fixed-4, fixed-8, fixed-16, fixed-32,
#                 adablock, ours-teacher, ours-oracle}     -> 7
# Total cells = 4 * 7 = 28. LLaDA-only.

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

mkdir -p "${EAI_RESULT_DIR}/base"

MODEL=llada
BENCHMARKS=(gsm8k math humaneval mbpp)
SCHEDULERS=(fixed-4 fixed-8 fixed-16 fixed-32 adablock ours-teacher ours-oracle)

IDX=$SLURM_ARRAY_TASK_ID
S_LEN=${#SCHEDULERS[@]}
B_LEN=${#BENCHMARKS[@]}

S_IDX=$(( IDX % S_LEN ))
IDX=$(( IDX / S_LEN ))
B_IDX=$(( IDX % B_LEN ))

BENCHMARK=${BENCHMARKS[$B_IDX]}
SCHEDULER=${SCHEDULERS[$S_IDX]}

PRED_ARG=""
if [[ "$SCHEDULER" == ours-teacher ]]; then
  PRED_ARG="--predictor ${EAI_CKPT_DIR}/base/${MODEL}_teacher.pt"
elif [[ "$SCHEDULER" == ours-oracle ]]; then
  PRED_ARG="--predictor ${EAI_CKPT_DIR}/base/${MODEL}_oracle.pt"
fi

OUT="${EAI_RESULT_DIR}/base/${MODEL}_${BENCHMARK}_${SCHEDULER}.json"

run_in_singularity "
  python -u -m src.eval.run_benchmarks \
    --model ${MODEL} --model_id GSAI-ML/LLaDA-8B-Base \
    --benchmark ${BENCHMARK} \
    --split test \
    --n_prompts 200 \
    --max_new_tokens 384 \
    --n_denoise_steps 32 \
    --min_new_tokens 128 \
    --scheduler ${SCHEDULER} ${PRED_ARG} \
    --lam 0.05 \
    --threshold 0.9 \
    --initial_block_size 16 \
    --out ${OUT} \
    --dtype bf16
"
