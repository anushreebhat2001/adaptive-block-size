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
#SBATCH --array=0-55%6

# Base-model eval (LLaDA-Base + Dream-Base). Reads predictor ckpts from
# ${EAI_CKPT_DIR}/base/ and writes results to ${EAI_RESULT_DIR}/base/.
# Instruct results from the prior pipeline run remain on disk untouched.
#
# Dropped ifeval: scoring.ifeval_correct returns len(generated)>0, which
# is always 1.0 and contributed the suspicious "ifeval=1.000 across all
# schedulers" line to the prior eval (no real signal).
#
# Array index decomposes into:
#   model      ∈ {llada, dream}                            -> 2
#   benchmark  ∈ {gsm8k, math, humaneval, mbpp}            -> 4
#   scheduler  ∈ {fixed-4, fixed-8, fixed-16, fixed-32,
#                 adablock, ours-teacher, ours-oracle}     -> 7
# Total cells = 2 * 4 * 7 = 56.
#
# EVAL_TAG (env var) selects which predictor variant to load. Default ""
# uses the base ckpts (e.g. llada_oracle.pt). To eval a lambda variant,
# submit with the appropriate suffix:
#   sbatch                                  slurm/04_eval_benchmarks.sbatch    # base
#   sbatch --export=ALL,EVAL_TAG=_lam5      slurm/04_eval_benchmarks.sbatch    # lam=0.05
#   sbatch --export=ALL,EVAL_TAG=_lam_10    slurm/04_eval_benchmarks.sbatch    # lam=0.10
#   sbatch --export=ALL,EVAL_TAG=_lam_15    slurm/04_eval_benchmarks.sbatch    # lam=0.15
#   sbatch --export=ALL,EVAL_TAG=_lam_20    slurm/04_eval_benchmarks.sbatch    # lam=0.20
# Result JSONs include the tag so they don't collide:
#   results/base/llada_gsm8k_ours-oracle_lam_15.json

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

: "${EVAL_TAG:=}"

mkdir -p "${EAI_RESULT_DIR}/base"

MODELS=(llada dream)
MODEL_IDS=(GSAI-ML/LLaDA-8B-Base Dream-org/Dream-v0-Base-7B)
BENCHMARKS=(gsm8k math humaneval mbpp)
SCHEDULERS=(fixed-4 fixed-8 fixed-16 fixed-32 adablock ours-teacher ours-oracle)

IDX=$SLURM_ARRAY_TASK_ID
S_LEN=${#SCHEDULERS[@]}
B_LEN=${#BENCHMARKS[@]}
M_LEN=${#MODELS[@]}

S_IDX=$(( IDX % S_LEN ))
IDX=$(( IDX / S_LEN ))
B_IDX=$(( IDX % B_LEN ))
IDX=$(( IDX / B_LEN ))
M_IDX=$(( IDX % M_LEN ))

MODEL=${MODELS[$M_IDX]}
MODEL_ID=${MODEL_IDS[$M_IDX]}
BENCHMARK=${BENCHMARKS[$B_IDX]}
SCHEDULER=${SCHEDULERS[$S_IDX]}

# When EVAL_TAG is set, only ours-oracle changes — teacher rule has no
# lambda parameter, and the fixed-*/adablock baselines don't use a
# predictor at all. Skip everything except ours-oracle to avoid
# re-running the baseline eval for every lambda sweep point.
if [[ -n "$EVAL_TAG" && "$SCHEDULER" != "ours-oracle" ]]; then
  echo "[eval] SKIP: scheduler=$SCHEDULER does not depend on EVAL_TAG ('$EVAL_TAG'); covered by the default (no-tag) base eval."
  exit 0
fi

PRED_ARG=""
PRED_PATH=""
if [[ "$SCHEDULER" == "ours-teacher" ]]; then
  # Teacher rule is lambda-free; always load the base teacher ckpt.
  PRED_PATH="${EAI_CKPT_DIR}/base/${MODEL}_teacher.pt"
elif [[ "$SCHEDULER" == "ours-oracle" ]]; then
  # Oracle predictor is the only one that varies with lambda.
  PRED_PATH="${EAI_CKPT_DIR}/base/${MODEL}_oracle${EVAL_TAG}.pt"
fi

# Final guard: if the chosen ckpt doesn't exist on disk, skip cleanly
# (instead of crashing the array task).
if [[ -n "$PRED_PATH" && ! -f "$PRED_PATH" ]]; then
  echo "[eval] SKIP: predictor not found at $PRED_PATH (model=$MODEL benchmark=$BENCHMARK scheduler=$SCHEDULER tag='$EVAL_TAG')"
  exit 0
fi
if [[ -n "$PRED_PATH" ]]; then
  PRED_ARG="--predictor $PRED_PATH"
fi

# Output filename: tag is included only for ours-oracle, since that's
# the only scheduler whose result changes with EVAL_TAG.
if [[ "$SCHEDULER" == "ours-oracle" ]]; then
  OUT="${EAI_RESULT_DIR}/base/${MODEL}_${BENCHMARK}_${SCHEDULER}${EVAL_TAG}.json"
else
  OUT="${EAI_RESULT_DIR}/base/${MODEL}_${BENCHMARK}_${SCHEDULER}.json"
fi

run_in_singularity "
  python -u -m src.eval.run_benchmarks \
    --model ${MODEL} --model_id ${MODEL_ID} \
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
