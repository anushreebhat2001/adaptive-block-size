"""Inspect class distribution of teacher and oracle label shards.

Reads every shard, aggregates per-benchmark and overall class counts,
and prints a clean table. Use this between label generation (01/02)
and predictor training (03) to verify the labels span all 4 classes.

Usage:
    python -m scripts.check_labels
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from typing import Dict, List

import torch


def parse_args() -> argparse.Namespace:
    user = os.environ.get("USER", "")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--label_dir",
        default=f"/scratch/{user}/Efficient-AI/labels",
    )
    p.add_argument("--model", default="llada")
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
    return (
        f"{name:<28} {n_total_shards:>3} shards  "
        f"N={total:>6}  "
        f"B=4:{counts[0]:>5} ({pcts[0]:>4.1f}%)  "
        f"B=8:{counts[1]:>5} ({pcts[1]:>4.1f}%)  "
        f"B=16:{counts[2]:>5} ({pcts[2]:>4.1f}%)  "
        f"B=32:{counts[3]:>5} ({pcts[3]:>4.1f}%)  "
        f"H={entropy:.3f}  "
        f"classes={nonzero}/4"
    )


def report(label_dir: str, source: str, model: str) -> None:
    print(f"\n=== {source} labels ===")
    base = os.path.join(label_dir, source, model)
    if not os.path.isdir(base):
        print(f"  not found: {base}")
        return
    benchmarks = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    overall_counts = [0, 0, 0, 0]
    overall_shards = 0
    for bm in benchmarks:
        shard_paths = sorted(glob.glob(os.path.join(base, bm, "*.pt")))
        counts = load_label_counts(shard_paths)
        print("  " + fmt_row(bm, counts, len(shard_paths)))
        for i in range(4):
            overall_counts[i] += counts[i]
        overall_shards += len(shard_paths)
    print("  " + "-" * 100)
    print("  " + fmt_row("ALL", overall_counts, overall_shards))


def main() -> None:
    args = parse_args()
    print(f"label_dir = {args.label_dir}")
    print(f"model     = {args.model}")
    report(args.label_dir, "teacher", args.model)
    report(args.label_dir, "oracle", args.model)


if __name__ == "__main__":
    main()
