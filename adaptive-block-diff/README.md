# adaptive-block-diff (v2)

Supervised block-size predictor for diffusion language models. Trained
once offline against AdaBlock-dLLM "teacher" labels and against K-rollout
"oracle" labels; deployed as a drop-in scheduler for LLaDA-8B and Dream-7B.

## Layout

```
adaptive-block-diff/
├── src/
│   ├── predictor/        # MLP, dataset, training
│   ├── data/             # diffusion runners, label builders, benchmark loaders
│   ├── inference/        # scheduler + scheduled sampler, optional fork patches
│   └── eval/             # run_benchmarks driver, scoring, plotting
├── slurm/                # 4 sbatch jobs
├── third_party/          # AdaBlock-dLLM submodule (optional)
└── requirements.txt
```

## HPC pipeline (~250 GPU-hours total)

The four sbatch jobs in `slurm/` correspond to the four pipeline stages.

```
01_build_teacher_labels.sbatch    array 0..7 (model x benchmark)   ~10h total
02_build_oracle_labels.sbatch     array 0..7                       ~80h total
03_train_predictor.sbatch         array 0..3                       ~4h total
04_eval_benchmarks.sbatch         array 0..69                      ~120h total
```

Override paths via env vars before sbatch (defaults match `_common.sh`):

```
export EAI_REPO=/scratch/$USER/adaptive-block-diff
export EAI_OVERLAY=/scratch/$USER/overlay-50G-10M.ext3
export EAI_CONDA_ENV=adaptive-block-diff
sbatch slurm/01_build_teacher_labels.sbatch
```

## Local setup (one-time, on HPC login node)

```
mkdir -p /scratch/$USER && cd /scratch/$USER
git clone <this-repo> adaptive-block-diff
cd adaptive-block-diff
# create overlay + conda env once
singularity exec --overlay /scratch/$USER/overlay-50G-10M.ext3:rw \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/miniconda3/bin/activate && \
    conda create -n adaptive-block-diff python=3.10 -y && \
    conda activate adaptive-block-diff && \
    pip install -r requirements.txt"
```

## Optional: AdaBlock-dLLM fork

For headline numbers reproduced against the exact AdaBlock-dLLM sampler:

```
git submodule add https://github.com/lgxi24/AdaBlock-dLLM third_party/AdaBlock-dLLM
git submodule update --init --recursive
```

`src/inference/llada_patch.py` and `dream_patch.py` are no-ops if the
submodule is missing; the rest of the pipeline runs against
`scheduled_sampler.py` instead.

## Verification gates (run in order)

1. **Pipeline smoke** — 100-prompt teacher-label run + 1-epoch train + 50-prompt eval. Predictor's chosen block sizes should match AdaBlock's choices within ~5%.
   ```
   python -m src.data.build_teacher_labels --model llada --benchmark gsm8k --n_prompts 100 \
       --out_dir /tmp/labels --shard_size 100 --max_new_tokens 128
   python -m src.predictor.train --shard_glob '/tmp/labels/teacher/llada/gsm8k/*.pt' \
       --out_ckpt /tmp/ckpt.pt --label_source teacher --model llada --hidden_dim 4096 --epochs 1
   python -m src.eval.run_benchmarks --model llada --benchmark gsm8k --scheduler ours-teacher \
       --predictor /tmp/ckpt.pt --n_prompts 50 --max_new_tokens 128 --out /tmp/smoke.json
   ```
2. **Oracle sanity** — oracle-label predictor val-loss ≤ teacher-label predictor val-loss.
3. **End-to-end** — full GSM8K eval (200 prompts) at our scheduler ≥ AdaBlock accuracy at ≥ AdaBlock throughput on at least one λ.
4. **Cost claim** — sum of GPU-hours for label gen + SL train < 50% of CtrlDiff's reported PPO budget.

## Plot

```
python -m src.eval.plot_pareto \
  --results_glob "/scratch/$USER/results/*.json" \
  --out_dir      /scratch/$USER/figures \
  --cost_json    /scratch/$USER/figures/cost.json
```
