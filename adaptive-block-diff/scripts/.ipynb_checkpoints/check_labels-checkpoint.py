"""Inspect class distribution of teacher and oracle label shards.

Reads every shard, aggregates per-benchmark and overall class counts,
and prints a clean table. Use this between label generation (01/02)
and predictor training (03) to verify the labels span all 4 classes.

Auto-discovers models present under ${label_dir}/{teacher,oracle}/.
The default label_dir points at the Base full-pipeline output dir.

Usage:
    python -m scripts.check_labels
    python -m scripts.check_labels --label_dir /scratch/$USER/Efficient-AI/labels/base
    python -m scripts.check_labels --model llada     # restrict to one model
    python -m scripts.check_labels --model llada,dream
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from typing import List, Optional

import torch


def parse_args() -> argparse.Namespace:
    user = os.environ.get("USER", "")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--label_dir",
        default=f"/scratch/{user}/Efficient-AI/labels/base",
        help="Root labels dir (expects teacher/, oracle/ subdirs each containing per-model subdirs).",
    )
    p.add_argument(
        "--model",
        default="auto",
        help="Comma-separated model names (e.g. 'llada,dream') or 'auto' to discover from disk.",
    )
    return p.parse_args()


def load_label_counts(shard_paths: List[str]) -> List[int]:
    if not shard_paths:
        return [0, 0, 0, 0]
    all_labels = torch.cat(
        [torch.load(p, map_location="cpu")["labels"] for p in shard_paths]
    )
    return torch.bincount(all_labels, minlength=4).tolist()


def fmt_row(name: str, counts: List[int], n_total_shards: int) -> str:
    total = sum(counts)
    if total == 0:
        return f"{name:<28} 0 shards"
    pcts = [100 * c / total for c in counts]
    probs = [c / total for c in counts]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    nonzero = sum(1 for c in counts if c > 0)
    max_pct = max(pcts)
    flag = ""
    if nonzero < 3:
        flag = " ⚠ <3 classes"
    elif max_pct > 70:
        flag = " ⚠ class >70%"
    return (
        f"{name:<28} {n_total_shards:>3} shards  "
        f"N={total:>6}  "
        f"B=4:{counts[0]:>5} ({pcts[0]:>4.1f}%)  "
        f"B=8:{counts[1]:>5} ({pcts[1]:>4.1f}%)  "
        f"B=16:{counts[2]:>5} ({pcts[2]:>4.1f}%)  "
        f"B=32:{counts[3]:>5} ({pcts[3]:>4.1f}%)  "
        f"entropy={entropy:.3f}  "
        f"max_class={max_pct:.1f}%  "
        f"classes={nonzero}/4{flag}"
    )


def discover_models(label_dir: str) -> List[str]:
    """Find which models have shards under label_dir/{teacher,oracle}/."""
    found = set()
    for source in ("teacher", "oracle"):
        base = os.path.join(label_dir, source)
        if not os.path.isdir(base):
            continue
        for d in os.listdir(base):
            if os.path.isdir(os.path.join(base, d)):
                found.add(d)
    return sorted(found)


def report(label_dir: str, source: str, model: str) -> Optional[List[int]]:
    """Print per-benchmark + ALL row for one (source, model). Returns ALL counts."""
    print(f"\n=== {source} / {model} ===")
    base = os.path.join(label_dir, source, model)
    if not os.path.isdir(base):
        print(f"  not found: {base}")
        return None
    benchmarks = sorted(
        d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))
    )
    if not benchmarks:
        print(f"  no benchmark subdirs under {base}")
        return None
    overall_counts = [0, 0, 0, 0]
    overall_shards = 0
    for bm in benchmarks:
        shard_paths = sorted(glob.glob(os.path.join(base, bm, "*.pt")))
        counts = load_label_counts(shard_paths)
        print("  " + fmt_row(bm, counts, len(shard_paths)))
        for i in range(4):
            overall_counts[i] += counts[i]
        overall_shards += len(shard_paths)
    print("  " + "-" * 110)
    print("  " + fmt_row("ALL", overall_counts, overall_shards))
    return overall_counts


def main() -> None:
    args = parse_args()
    print(f"label_dir = {args.label_dir}")

    if args.model == "auto":
        models = discover_models(args.label_dir)
        if not models:
            print(f"No models discovered under {args.label_dir}/. "
                  f"Pass --model explicitly or check the path.")
            return
        print(f"models    = {models} (auto-discovered)")
    else:
        models = [m.strip() for m in args.model.split(",") if m.strip()]
        print(f"models    = {models}")

    summary_rows = []  # (source, model, total_N, classes_hit, entropy, max_pct)
    for source in ("teacher", "oracle"):
        for model in models:
            counts = report(args.label_dir, source, model)
            if counts is None:
                continue
            n = sum(counts)
            if n == 0:
                continue
            probs = [c / n for c in counts]
            H = -sum(p * math.log(p) for p in probs if p > 0)
            nz = sum(1 for c in counts if c > 0)
            max_pct = max(100 * c / n for c in counts)
            summary_rows.append((source, model, n, nz, H, max_pct))

    if summary_rows:
        print("\n" + "=" * 110)
        print("CROSS-MODEL SUMMARY (ALL rows)")
        print("=" * 110)
        print(f"{'source':<10} {'model':<10} {'N':>8} {'classes':>8} "
              f"{'entropy':>9} {'max_class%':>11}  status")
        print("-" * 110)
        for source, model, n, nz, H, max_pct in summary_rows:
            status = "ok"
            if nz < 3:
                status = "⚠ <3 classes"
            elif max_pct > 70:
                status = "⚠ max-class >70%"
            elif H < 1.0:
                status = "⚠ low entropy"
            print(f"{source:<10} {model:<10} {n:>8} {nz:>6}/4 "
                  f"{H:>9.3f} {max_pct:>10.1f}%  {status}")


if __name__ == "__main__":
    main()
