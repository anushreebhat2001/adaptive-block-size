#!/bin/bash
#SBATCH --job-name=adb_smoke_dream_base
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.err

# Dream-7B-Base smoke. Uses Dream-org/Dream-v0-Base-7B (no chat template,
# raw text continuation -- analog of LLaDA-Base). Output goes to
# ${EAI_SMOKE_DIR}/dream_base/ alongside the other smokes.
#
# Hypothesis: same as the LLaDA-Base story -- Instruct variant bails after
# one block of EOS, Base variant continues raw text and gives us real
# delimiter signals + 4-class teacher labels.
#
# If this passes, Dream-Base is the deployable Dream variant for the
# paper, not Dream-Instruct.

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

SMOKE_DIR=${EAI_SMOKE_DIR}/dream_base
mkdir -p "${SMOKE_DIR}/labels" "${SMOKE_DIR}/ckpt" "${SMOKE_DIR}/results"

run_in_singularity "
  set -euo pipefail

  echo '[smoke-dream-base] step 1/3: build_teacher_labels (Dream-Base, GSM8K, debug first 30 boundaries)'
  python -u -m src.data.build_teacher_labels \
    --model dream --model_id Dream-org/Dream-v0-Base-7B \
    --benchmark gsm8k \
    --split train --n_prompts 25 \
    --out_dir ${SMOKE_DIR}/labels \
    --shard_size 1000 \
    --max_new_tokens 256 \
    --n_denoise_steps 32 \
    --default_block_size 16 \
    --min_new_tokens 128 \
    --debug_n_boundaries 30 \
    --dtype bf16

  echo '[smoke-dream-base] step 2/3: train predictor (5 epochs)'
  python -u -m src.predictor.train \
    --shard_glob '${SMOKE_DIR}/labels/dream/gsm8k/*.pt' \
    --out_ckpt   ${SMOKE_DIR}/ckpt/dream_teacher.pt \
    --label_source teacher --model dream \
    --hidden_dim 3584 \
    --epochs 5 --batch_size 256 --val_frac 0.2 \
    --class_weight_mode none --label_smoothing 0.1

  echo '[smoke-dream-base] step 3/3: eval ours-teacher vs adablock on 15 GSM8K prompts'
  python -u -m src.eval.run_benchmarks \
    --model dream --model_id Dream-org/Dream-v0-Base-7B \
    --benchmark gsm8k \
    --split test --n_prompts 15 \
    --max_new_tokens 256 --n_denoise_steps 32 \
    --min_new_tokens 128 \
    --scheduler ours-teacher \
    --predictor ${SMOKE_DIR}/ckpt/dream_teacher.pt \
    --initial_block_size 16 \
    --out ${SMOKE_DIR}/results/ours_teacher.json \
    --dtype bf16

  python -u -m src.eval.run_benchmarks \
    --model dream --model_id Dream-org/Dream-v0-Base-7B \
    --benchmark gsm8k \
    --split test --n_prompts 15 \
    --max_new_tokens 256 --n_denoise_steps 32 \
    --min_new_tokens 128 \
    --scheduler adablock \
    --initial_block_size 16 \
    --out ${SMOKE_DIR}/results/adablock.json \
    --dtype bf16

  echo '[smoke-dream-base] done.'
  echo '  Pass criteria (compare to LLaDA-Base smoke):'
  echo '    1. Decoded blocks show real reasoning text (not <|endoftext|>)'
  echo '    2. Boundaries-per-prompt >= 6 (buf count after 25 prompts >= 150)'
  echo '    3. Teacher class distribution covers >= 3 classes, no class > 70%'
  echo '    4. Eval block_size_histogram differs between ours-teacher and adablock'
  echo '  If all pass: Dream-Base is the second model in the paper.'
"
