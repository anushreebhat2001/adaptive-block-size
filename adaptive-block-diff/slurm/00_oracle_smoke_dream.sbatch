#!/bin/bash
#SBATCH --job-name=adb_oracle_smoke_dream
#SBATCH --account=ece_gy9483-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.out
#SBATCH --error=/scratch/%u/Efficient-AI/adaptive-block-diff/slurm/logs/%x_%j.err

# Dream-7B-Instruct oracle smoke. Uses default model_id (Dream-org/
# Dream-v0-Instruct-7B). Output goes to ${EAI_SMOKE_DIR}/dream/oracle/.
#
# Run AFTER 00_smoke_dream.sbatch produces healthy teacher labels.
# Prior pre-fix smoke showed Dream oracle labels collapsing to 2 classes
# (uniform PPL across L), which is why Dream was excluded from the full
# pipeline. This re-checks under the current configuration with
# --min_new_tokens=128 backstop and the chat-template fix in encode_prompt.
#
# Hidden dim is 3584 for Dream (vs 4096 for LLaDA).

source /scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/_common.sh

SMOKE_DIR=${EAI_SMOKE_DIR}/dream/oracle
mkdir -p "${SMOKE_DIR}/labels" "${SMOKE_DIR}/ckpt" "${SMOKE_DIR}/results"

run_in_singularity "
  set -euo pipefail

  echo '[oracle-smoke-dream] step 1/3: build_oracle_labels (Dream-Instruct, 12 GSM8K prompts x K=4 rollouts)'
  python -u -m src.data.build_oracle_labels \
    --model dream \
    --benchmark gsm8k \
    --split train --n_prompts 12 \
    --out_dir ${SMOKE_DIR}/labels \
    --shard_size 1000 \
    --max_new_tokens 256 \
    --n_denoise_steps 32 \
    --default_block_size 16 \
    --min_new_tokens 128 \
    --lam 0.05 \
    --dtype bf16

  echo '[oracle-smoke-dream] step 2/3: train predictor on oracle labels (5 epochs)'
  python -u -m src.predictor.train \
    --shard_glob '${SMOKE_DIR}/labels/dream/gsm8k/*.pt' \
    --out_ckpt   ${SMOKE_DIR}/ckpt/dream_oracle.pt \
    --label_source oracle --model dream \
    --hidden_dim 3584 \
    --epochs 5 --batch_size 256 --val_frac 0.2 \
    --class_weight_mode none --label_smoothing 0.1

  echo '[oracle-smoke-dream] step 3/3: eval ours-oracle vs adablock on 15 GSM8K prompts'
  python -u -m src.eval.run_benchmarks \
    --model dream \
    --benchmark gsm8k \
    --split test --n_prompts 15 \
    --max_new_tokens 256 --n_denoise_steps 32 \
    --min_new_tokens 128 \
    --scheduler ours-oracle \
    --predictor ${SMOKE_DIR}/ckpt/dream_oracle.pt \
    --initial_block_size 16 \
    --out ${SMOKE_DIR}/results/ours_oracle.json \
    --dtype bf16

  python -u -m src.eval.run_benchmarks \
    --model dream \
    --benchmark gsm8k \
    --split test --n_prompts 15 \
    --max_new_tokens 256 --n_denoise_steps 32 \
    --min_new_tokens 128 \
    --scheduler adablock \
    --initial_block_size 16 \
    --out ${SMOKE_DIR}/results/adablock.json \
    --dtype bf16

  echo '[oracle-smoke-dream] done.'
  echo '  GO criteria for adding Dream to the full pipeline:'
  echo '    1. Oracle label class_balance covers >=3 classes, no class >70%'
  echo '    2. Predictor val gain_over_majority > 0'
  echo '    3. Eval block_size_histogram differs visibly between schedulers'
  echo '  NO-GO: oracle labels still collapse to <3 classes (matches prior result)'
  echo '         => stick with LLaDA-only, drop Dream from paper or relegate to limitation'
"
