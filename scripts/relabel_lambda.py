"""Relabel oracle shards at new lambda values without re-running K-rollouts.

Reads existing oracle shards (which cache `per_l_ppl: [N, K]`), recomputes
labels at user-supplied lambda values via

    label_n = argmax_L  ( -per_l_ppl[n, L_idx]  -  lambda * (1 / L) )

and saves copies of each shard under a per-lambda subtree:

    <out_dir>/lam_<value>/<source>/<model>/<benchmark>/<filename>

The copies preserve `scalars`, `hidden_pool`, `prompt_ids`, and `per_l_ppl`
so the existing PredictorDataset / training pipeline can consume them with
no code changes - just point `--shard_glob` at the new tree.

Per-shard cost is dominated by I/O (load + write); the relabel itself is
a single argmax over [N, 4]. Expect ~30s-2min for the full oracle tree
(86 shards x 2 models x 2 lambdas) on HPC scratch.

Example:

    python -m scripts.relabel_lambda \
        --in_dir  /scratch/$USER/Efficient-AI/labels/base \
        --out_dir /scratch/$USER/Efficient-AI/labels \
        --lambdas 0.01,1.0

Then to train at, say, lambda=1.0:

    python -m src.predictor.train \
        --shard_glob '/scratch/$USER/Efficient-AI/labels/lam_1/oracle/llada/gsm8k/*.pt' \
        --out_ckpt   /scratch/$USER/Efficient-AI/ckpts/llada_oracle_lam1.pt \
        --label_source oracle --model llada \
        --hidden_dim 4096 --epochs 16
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from collections import Counter
from typing import Dict, List

import torch


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
        help="Root under which lam_<value>/<source>/<model>/<benchmark>/ "
             "subtrees will be written.",
    )
    p.add_argument(
        "--lambdas",
        default="0.01,1.0",
        help="Comma-separated lambda values to relabel at.",
    )
    p.add_argument(
        "--source",
        default="oracle",
        help="Top-level source subdir under --in_dir to scan (default: oracle).",
    )
    p.add_argument(
        "--shard_prefix",
        default="oracle",
        help="Filename prefix for shards (default: oracle).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Scan and report distributions without writing any output files.",
    )
    return p.parse_args()


def _lam_tag(lam: float) -> str:
    """Filesystem-safe tag for a lambda value: 0.01 -> lam_0p01, 1.0 -> lam_1."""
    s = f"{lam:g}"  # compact float repr; 0.01 -> '0.01', 1.0 -> '1'
    return "lam_" + s.replace(".", "p")


def relabel_per_l_ppl(
    per_l_ppl: torch.Tensor,
    lam: float,
    candidate_block_sizes: List[int] = CANDIDATE_BLOCK_SIZES,
) -> torch.Tensor:
    """per_l_ppl: [N, K]. Returns LongTensor [N] of argmax indices into K."""
    if per_l_ppl.ndim != 2 or per_l_ppl.shape[-1] != len(candidate_block_sizes):
        raise ValueError(
            f"expected per_l_ppl shape [N, {len(candidate_block_sizes)}], "
            f"got {tuple(per_l_ppl.shape)}"
        )
    l_inv = torch.tensor(
        [1.0 / L for L in candidate_block_sizes], dtype=torch.float32
    )  # [K]
    utilities = -per_l_ppl.to(torch.float32) - lam * l_inv.unsqueeze(0)  # [N, K]
    return utilities.argmax(dim=-1).to(torch.long)


def main() -> None:
    args = parse_args()
    lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    if not lambdas:
        raise SystemExit("no lambda values parsed from --lambdas")

    pattern = os.path.join(
        args.in_dir, args.source, "*", "*", f"{args.shard_prefix}_*.pt"
    )
    shard_paths = sorted(glob.glob(pattern))
    if not shard_paths:
        raise FileNotFoundError(f"no shards matched: {pattern}")

    print(f"[relabel] found {len(shard_paths)} shards under "
          f"{args.in_dir}/{args.source}")
    print(f"[relabel] relabeling at lambdas={lambdas}")
    if args.dry_run:
        print("[relabel] DRY RUN - not writing any files")

    # Per-(lam, model, benchmark) histograms for end-of-run report.
    per_key_hist: Dict[str, Counter] = {}

    t0 = time.time()
    n_examples_total = 0
    bytes_written = 0

    for i, sp in enumerate(shard_paths):
        rel = os.path.relpath(sp, os.path.join(args.in_dir, args.source))
        parts = rel.split(os.sep)
        if len(parts) < 3:
            print(f"[relabel] skipping {sp} (unexpected path layout)")
            continue
        model, benchmark, fname = parts[0], parts[1], parts[-1]

        try:
            shard = torch.load(sp, map_location="cpu")
        except Exception as e:
            print(f"[relabel] failed to load {sp}: {e}")
            continue

        if "per_l_ppl" not in shard:
            print(f"[relabel] skipping {sp} (no per_l_ppl - not an oracle shard?)")
            continue

        per_l_ppl = shard["per_l_ppl"]
        n_examples_total += per_l_ppl.shape[0]

        for lam in lambdas:
            new_labels = relabel_per_l_ppl(per_l_ppl, lam)

            key = f"{lam:g}|{model}|{benchmark}"
            hist = per_key_hist.setdefault(key, Counter())
            for v in new_labels.tolist():
                hist[int(v)] += 1

            if args.dry_run:
                continue

            out_subdir = os.path.join(
                args.out_dir, _lam_tag(lam), args.source, model, benchmark
            )
            os.makedirs(out_subdir, exist_ok=True)
            out_path = os.path.join(out_subdir, fname)

            new_shard = dict(shard)  # shallow copy of top-level keys
            new_shard["labels"] = new_labels
            new_meta = dict(shard.get("meta", {}))
            new_meta["lambda"] = lam
            new_meta["relabeled_from"] = sp
            new_shard["meta"] = new_meta

            torch.save(new_shard, out_path)
            try:
                bytes_written += os.path.getsize(out_path)
            except OSError:
                pass

        if (i + 1) % 20 == 0 or (i + 1) == len(shard_paths):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            print(
                f"[relabel] {i+1}/{len(shard_paths)} shards processed "
                f"({rate:.1f} shards/s, {elapsed:.1f}s elapsed)"
            )

    elapsed = time.time() - t0
    print(
        f"\n[relabel] done in {elapsed:.1f}s. "
        f"shards={len(shard_paths)} lambdas={len(lambdas)} "
        f"examples_per_lambda={n_examples_total} "
        f"bytes_written={bytes_written/1e9:.2f}GB"
    )

    print("\n=== relabel summary (per lambda x model x benchmark) ===")
    for lam in lambdas:
        print(f"\n  lambda = {lam}")
        rows = sorted(k for k in per_key_hist if k.startswith(f"{lam:g}|"))
        for key in rows:
            _, model, benchmark = key.split("|")
            c = per_key_hist[key]
            total = sum(c.values())
            n_classes = sum(
                1 for k in range(len(CANDIDATE_BLOCK_SIZES)) if c.get(k, 0) > 0
            )
            counts_str = " ".join(
                f"B={CANDIDATE_BLOCK_SIZES[k]}:{c.get(k,0):5d}({100*c.get(k,0)/max(total,1):4.1f}%)"
                for k in range(len(CANDIDATE_BLOCK_SIZES))
            )
            print(
                f"    {model:<8} {benchmark:<10} N={total:6d}  {counts_str}  "
                f"classes={n_classes}/{len(CANDIDATE_BLOCK_SIZES)}"
            )


if __name__ == "__main__":
    main()
