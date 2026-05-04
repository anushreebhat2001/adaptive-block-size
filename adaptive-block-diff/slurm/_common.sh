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
# :ro lets multiple array tasks mount the same overlay simultaneously.
# NYU HPC blocks concurrent :rw mounts of one overlay, so :rw would
# serialize the whole job array. Our pipeline reads the conda env from
# the overlay but writes only to bind-mounted /scratch dirs, so :ro is
# correct. Override to :rw via EAI_OVERLAY_MODE only when you need to
# install packages or modify the env mid-job (e.g. interactively).
: "${EAI_OVERLAY_MODE:=ro}"

mkdir -p "$EAI_LABEL_DIR" "$EAI_CKPT_DIR" "$EAI_RESULT_DIR" "$EAI_SLURM_LOGS"

run_in_singularity() {
  local cmd="$1"
  singularity exec --nv \
    --overlay "${EAI_OVERLAY}:${EAI_OVERLAY_MODE}" \
    "$EAI_SIF" \
    /bin/bash -c "
      source /ext3/miniconda3/bin/activate ${EAI_CONDA_ENV}
      export HF_HOME=${EAI_HF_HOME}
      # If you hit \"Stale file handle\" / FileLock errors on parallel
      # array jobs, set HF_DATASETS_OFFLINE=1 in your sbatch (the cache
      # must already be populated). Don't set it globally here -- it
      # would break first-time dataset downloads.
      export PYTHONPATH=${EAI_REPO}:\${PYTHONPATH:-}
      cd ${EAI_REPO}
      ${cmd}
    "
}
