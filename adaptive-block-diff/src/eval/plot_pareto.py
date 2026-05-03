"""Aggregate eval JSONs into the headline Pareto plot.

Reads a directory of run_benchmarks.py outputs and emits:
  - pareto_<benchmark>_<model>.png : accuracy vs tokens/sec, one curve per
    scheduler family (fixed schedulers as a single connected line, adablock
    as scatter with threshold sweep, ours-* as scatter with lambda sweep).
  - cost_summary.png : GPU-hours-to-train (label gen + SL train) bar chart
    with CtrlDiff cited cost as reference.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_glob", required=True, help="glob over result JSONs")
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--cost_json",
        default=None,
        help="optional JSON with GPU-hours per stage for the cost-summary plot",
    )
    return p.parse_args()


def _load(results_glob: str) -> List[Dict]:
    return [json.load(open(p)) for p in sorted(glob.glob(results_glob))]


def _color_for(scheduler: str) -> str:
    if scheduler.startswith("fixed"):
        return "tab:gray"
    if scheduler == "adablock":
        return "tab:orange"
    if scheduler.startswith("ours-oracle"):
        return "tab:red"
    if scheduler.startswith("ours-teacher"):
        return "tab:blue"
    return "tab:green"


def _plot_pareto(runs: List[Dict], benchmark: str, model: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    by_sched: Dict[str, List[Dict]] = defaultdict(list)
    for r in runs:
        if r["benchmark"] != benchmark or r["model"] != model:
            continue
        by_sched[r["scheduler"]].append(r)

    for sched, rs in by_sched.items():
        xs = [r["tokens_per_second"] for r in rs]
        ys = [r["accuracy"] for r in rs]
        if sched.startswith("fixed-"):
            order = sorted(range(len(rs)), key=lambda i: int(rs[i]["scheduler"].split("-")[1]))
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            ax.plot(xs, ys, marker="o", linestyle="-", color=_color_for(sched), label="fixed")
        else:
            ax.scatter(xs, ys, color=_color_for(sched), label=sched, s=40)

    ax.set_xlabel("tokens / second")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{model} / {benchmark}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_cost(cost_json: str, out_path: str) -> None:
    cost = json.load(open(cost_json))
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = list(cost.keys())
    vals = [cost[k] for k in labels]
    ax.bar(labels, vals)
    ax.set_ylabel("GPU-hours")
    ax.set_title("Cost to produce a working scheduler")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    runs = _load(args.results_glob)
    if not runs:
        print(f"no results matched {args.results_glob}")
        return

    pairs = sorted({(r["model"], r["benchmark"]) for r in runs})
    for model, benchmark in pairs:
        out_path = os.path.join(args.out_dir, f"pareto_{model}_{benchmark}.png")
        _plot_pareto(runs, benchmark, model, out_path)
        print(f"wrote {out_path}")

    if args.cost_json and os.path.exists(args.cost_json):
        out_path = os.path.join(args.out_dir, "cost_summary.png")
        _plot_cost(args.cost_json, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
