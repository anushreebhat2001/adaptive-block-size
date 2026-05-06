"""Plot oracle achievable-PPL vs achievable-compute Pareto curves for each
model across a fine lambda sweep, using cached per-L PPL from existing
oracle shards (no re-rollouts, no shard writes).

For each (model, lambda) pair, computes:
    label_n              = argmax_L  (-per_l_ppl[n, L_idx] - lambda * 1/L)
    achievable_PPL(λ)    = mean_n  per_l_ppl[n, label_n]
    achievable_compute(λ)= mean_n  1/L[label_n]    (lower = fewer forward
                                                    passes per token)

Outputs (under --out_dir):
    pareto_data.json      raw (lambda, ppl, compute) lists per model
    pareto_<model>.png    one curve per model (LLaDA, Dream)
    pareto_combined.png   all models side-by-side (if >=2 models present)

This script intentionally does NOT overlay AdaBlock or fixed-B baselines
- it shows only the oracle's own quality-vs-compute trade as lambda varies.

Example:
    python -m scripts.plot_pareto_oracle \
        --in_dir   /scratch/$USER/Efficient-AI/labels/base \
        --out_dir  /scratch/$USER/Efficient-AI/figures \
        --lambda_min 0.5 --lambda_max 50 --lambda_step 0.5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import defaultdict
from typing import Dict, List

import torch

import matplotlib
matplotlib.use("Agg")  # non-interactive; works on HPC login nodes
import matplotlib.pyplot as plt


CANDIDATE_BLOCK_SIZES: List[int] = [4, 8, 16, 32]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in_dir",
        required=True,
        help="Root containing oracle shards. Expected layout: "
             "<in_dir>/<source>/<model>/<benchmark>/<prefix>_*.pt",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Directory to save PNG plots and JSON data.",
    )
    p.add_argument("--source", default="oracle")
    p.add_argument("--shard_prefix", default="oracle")
    p.add_argument("--lambda_min", type=float, default=0.5,
                   help="Smallest lambda in the regular sweep.")
    p.add_argument("--lambda_max", type=float, default=50.0,
                   help="Largest lambda in the regular sweep (inclusive).")
    p.add_argument("--lambda_step", type=float, default=0.5,
                   help="Lambda increment for the regular sweep.")
    p.add_argument(
        "--include_small",
        default="0.01,0.05,0.1",
        help="Extra small lambdas to pin the high-quality end of the curve. "
             "Comma-separated. Set to empty string to skip.",
    )
    p.add_argument(
        "--annotate_lambdas",
        default="0.5,1,5,10,50",
        help="Lambdas at which to annotate the points on the plot.",
    )
    return p.parse_args()


def build_lambda_list(args: argparse.Namespace) -> List[float]:
    lambdas: List[float] = []
    if args.include_small:
        lambdas.extend(
            float(x) for x in args.include_small.split(",") if x.strip()
        )
    n_steps = int(round((args.lambda_max - args.lambda_min) / args.lambda_step)) + 1
    for i in range(n_steps):
        lambdas.append(args.lambda_min + i * args.lambda_step)
    lambdas = sorted({round(x, 6) for x in lambdas})
    return lambdas


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    lambdas = build_lambda_list(args)
    print(f"[plot] sweeping {len(lambdas)} lambda values "
          f"from {min(lambdas)} to {max(lambdas)}")

    pattern = os.path.join(
        args.in_dir, args.source, "*", "*", f"{args.shard_prefix}_*.pt"
    )
    shard_paths = sorted(glob.glob(pattern))
    if not shard_paths:
        raise FileNotFoundError(f"no shards matched: {pattern}")
    print(f"[plot] found {len(shard_paths)} oracle shards")

    l_inv = torch.tensor(
        [1.0 / L for L in CANDIDATE_BLOCK_SIZES], dtype=torch.float32
    )

    # accum[model][lambda] = [sum_ppl, sum_compute, n]
    accum: Dict[str, Dict[float, List[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0, 0])
    )

    t0 = time.time()
    for i, sp in enumerate(shard_paths):
        rel = os.path.relpath(sp, os.path.join(args.in_dir, args.source))
        parts = rel.split(os.sep)
        if len(parts) < 3:
            continue
        model = parts[0]

        try:
            shard = torch.load(sp, map_location="cpu")
        except Exception as e:
            print(f"[plot] failed to load {sp}: {e}")
            continue
        if "per_l_ppl" not in shard:
            continue
        per_l_ppl = shard["per_l_ppl"].to(torch.float32)  # [N, K]

        # vectorize across lambdas: utilities[L_idx_of_lambda, N, K]
        # but to keep memory bounded, loop.
        for lam in lambdas:
            utilities = -per_l_ppl - lam * l_inv.unsqueeze(0)  # [N, K]
            label_idx = utilities.argmax(dim=-1)               # [N]
            ppl_row = per_l_ppl.gather(1, label_idx.unsqueeze(-1)).squeeze(-1)
            cmp_row = l_inv[label_idx]
            finite = torch.isfinite(ppl_row)
            if finite.any():
                accum[model][lam][0] += float(ppl_row[finite].sum().item())
                accum[model][lam][1] += float(cmp_row[finite].sum().item())
                accum[model][lam][2] += int(finite.sum().item())

        if (i + 1) % 20 == 0 or (i + 1) == len(shard_paths):
            elapsed = time.time() - t0
            print(f"[plot] {i+1}/{len(shard_paths)} shards processed "
                  f"({elapsed:.1f}s elapsed)")

    print(f"[plot] sweep done in {time.time() - t0:.1f}s")

    # Aggregate
    pareto_data: Dict[str, Dict[str, List[float]]] = {}
    for model in sorted(accum.keys()):
        rows = []
        for lam in sorted(accum[model].keys()):
            sum_ppl, sum_cmp, n = accum[model][lam]
            if n > 0:
                rows.append((lam, sum_ppl / n, sum_cmp / n, n))
        pareto_data[model] = {
            "lambda":  [r[0] for r in rows],
            "ppl":     [r[1] for r in rows],
            "compute": [r[2] for r in rows],
            "n":       [r[3] for r in rows],
        }

    json_path = os.path.join(args.out_dir, "pareto_data.json")
    with open(json_path, "w") as f:
        json.dump(pareto_data, f, indent=2)
    print(f"[plot] wrote raw data to {json_path}")

    # Annotation lambdas
    annotate = []
    if args.annotate_lambdas:
        annotate = [
            float(x) for x in args.annotate_lambdas.split(",") if x.strip()
        ]

    def _draw_one(ax: plt.Axes, model: str, d: Dict[str, List[float]]) -> None:
        compute = d["compute"]
        ppl = d["ppl"]
        lam_arr = d["lambda"]
        ax.plot(compute, ppl, "-", color="0.6", linewidth=1, alpha=0.7, zorder=1)
        sc = ax.scatter(
            compute, ppl, c=lam_arr, cmap="viridis", s=28, zorder=3,
            edgecolors="white", linewidths=0.4,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("lambda (compute penalty weight)")
        for tgt in annotate:
            if tgt in lam_arr:
                idx = lam_arr.index(tgt)
                ax.annotate(
                    f"λ={tgt:g}",
                    (compute[idx], ppl[idx]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=9,
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="0.7", alpha=0.85),
                )
        # mark first and last lambda
        ax.scatter(
            [compute[0], compute[-1]], [ppl[0], ppl[-1]],
            facecolors="none", edgecolors="red", s=80, zorder=4,
            linewidths=1.2,
        )
        ax.annotate(
            f"smallest λ={lam_arr[0]:g}\n(quality-first)",
            (compute[0], ppl[0]), xytext=(10, -20),
            textcoords="offset points", fontsize=8, color="red",
        )
        ax.annotate(
            f"largest λ={lam_arr[-1]:g}\n(speed-first)",
            (compute[-1], ppl[-1]), xytext=(-90, 10),
            textcoords="offset points", fontsize=8, color="red",
        )
        ax.set_xlabel("achievable compute  (mean 1/L; lower = faster)")
        ax.set_ylabel("achievable PPL  (lower = better quality)")
        ax.set_title(f"Oracle Pareto curve - {model.upper()}")
        ax.grid(alpha=0.3)

    for model, d in pareto_data.items():
        fig, ax = plt.subplots(figsize=(8.5, 6))
        _draw_one(ax, model, d)
        fig.suptitle(
            "Bottom-left corner is ideal (high quality + low compute)",
            fontsize=10, y=0.995, color="0.3",
        )
        plt.tight_layout()
        path = os.path.join(args.out_dir, f"pareto_{model}.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[plot] wrote {path}")

    if len(pareto_data) >= 2:
        models_sorted = sorted(pareto_data.keys())
        fig, axes = plt.subplots(
            1, len(models_sorted), figsize=(8.5 * len(models_sorted), 6.5)
        )
        if len(models_sorted) == 1:
            axes = [axes]
        for ax, model in zip(axes, models_sorted):
            _draw_one(ax, model, pareto_data[model])
        fig.suptitle(
            "Oracle Pareto curves across lambda sweep "
            "(bottom-left = ideal: low PPL + low compute)",
            fontsize=12,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        path = os.path.join(args.out_dir, "pareto_combined.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[plot] wrote {path}")

    # Print a small text summary so the run is self-contained
    print("\n=== summary ===")
    for model, d in pareto_data.items():
        print(f"\n{model.upper()}:  {len(d['lambda'])} lambda points")
        # corners
        print(f"  smallest lambda={d['lambda'][0]:g}: "
              f"PPL={d['ppl'][0]:.4f} compute={d['compute'][0]:.5f}")
        print(f"  largest  lambda={d['lambda'][-1]:g}: "
              f"PPL={d['ppl'][-1]:.4f} compute={d['compute'][-1]:.5f}")
        # best on each axis
        best_q = min(range(len(d['ppl'])), key=lambda i: d['ppl'][i])
        best_s = min(range(len(d['compute'])), key=lambda i: d['compute'][i])
        print(f"  best PPL    : lambda={d['lambda'][best_q]:g} -> "
              f"PPL={d['ppl'][best_q]:.4f} compute={d['compute'][best_q]:.5f}")
        print(f"  best compute: lambda={d['lambda'][best_s]:g} -> "
              f"PPL={d['ppl'][best_s]:.4f} compute={d['compute'][best_s]:.5f}")

    print("[plot] done")


if __name__ == "__main__":
    main()
