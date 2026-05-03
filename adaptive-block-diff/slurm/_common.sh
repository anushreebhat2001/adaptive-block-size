# Sourced by every sbatch script. Override per-job via env vars.

set -euo pipefail

: "${EAI_REPO:=/scratch/$USER/Efficient-AI/adaptive-block-diff}"
: "${EAI_OVERLAY:=/scratch/$USER/overlay-50G-10M.ext3}"
: "${EAI_SIF:=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
: "${EAI_CONDA_ENV:=bd3lm}"
: "${EAI_HF_HOME:=/scratch/$USER/Efficient-AI/hf_cache}"
: "${EAI_LABEL_DIR:=/scratch/$USER/Efficient-AI/labels}"
: "${EAI_CKPT_DIR:=/scratch/$USER/Efficient-AI/ckpts/predictor}"
: "${EAI_RESULT_DIR:=/scratch/$USER/Efficient-AI/results}"
: "${EAI_SLURM_LOGS:=/scratch/$USER/Efficient-AI/adaptive-block-diff/slurm/logs}"
: "${EAI_SMOKE_DIR:=/scratch/$USER/Efficient-AI/smoke}"

mkdir -p "$EAI_LABEL_DIR" "$EAI_CKPT_DIR" "$EAI_RESULT_DIR" "$EAI_SLURM_LOGS"

run_in_singularity() {
  local cmd="$1"
  singularity exec --nv \
    --overlay "${EAI_OVERLAY}:rw" \
    "$EAI_SIF" \
    /bin/bash -c "
      source /ext3/miniconda3/bin/activate ${EAI_CONDA_ENV}
      export HF_HOME=${EAI_HF_HOME}
      export PYTHONPATH=${EAI_REPO}:\${PYTHONPATH:-}
      cd ${EAI_REPO}
      ${cmd}
    "
}
