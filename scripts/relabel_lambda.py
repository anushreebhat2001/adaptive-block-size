"""Relabel oracle shards at new lambda values without re-running K-rollouts,
then compute achievable (PPL, compute) Pareto points for each lambda and
emit a recommendation against fixed-B and AdaBlock-rule baselines.

Reads existing oracle shards (which cache `per_l_ppl: [N, K]`), recomputes
labels at user-supplied lambda values via

    label_n = argmax_L  ( -per_l_ppl[n, L_idx]  -  lambda * (1 / L) )

and saves copies of each shard under a per-lambda subtree:

    <out_dir>/lam_<value>/<source>/<model>/<benchmark>/<filename>

The copies preserve `scalars`, `hidden_pool`, `prompt_ids`, and `per_l_ppl`
so the existing PredictorDataset / training pipeline can consume them with
no code changes - just point `--shard_glob` at the new tree.

After relabeling, the script computes the *achievable* Pareto point at
each lambda:

    achievable_ppl(lambda)     = mean_n  per_l_ppl[n, label_n]
    achievable_compute(lambda) = mean_n  1 / L[label_n]    (proportional
                                  to forward passes per token; lower=faster)

These are upper bounds on what training a predictor at that lambda can
achieve. The script also computes the same point for each fixed-B
baseline (always picking the same L) and for AdaBlock-rule labels if a
matching teacher tree exists under <in_dir>/teacher/.

Finally a per-model verdict is printed identifying the lambda that
Pareto-dominates AdaBlock (lower PPL AND lower compute), or, failing
that, the lambda nearest AdaBlock's operating point.

Example:

    python -m scripts.relabel_lambda \
        --in_dir   /scratch/$USER/Efficient-AI/labels/base \
        --out_dir  /scratch/$USER/Efficient-AI/labels \
        --lambdas  0.01,0.05,0.1,0.5,1.0,5.0
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch


CANDIDATE_BLOCK_SIZES: List[int] = [4, 8, 16, 32]


# ------------------------------------------------------------------------- #
# argument parsing
# ------------------------------------------------------------------------- #


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
        default="0.01,0.05,0.1,0.5,1.0,5.0",
        help="Comma-separated lambda values to relabel and score.",
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
        "--teacher_source",
        default="teacher",
        help="Source subdir for teacher (AdaBlock-rule) shards if available, "
             "used as a reference Pareto point. Set to '' to skip.",
    )
    p.add_argument(
        "--teacher_prefix",
        default="teacher",
        help="Filename prefix for teacher shards.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute Pareto points and verdict without writing any files.",
    )
    return p.parse_args()


def _lam_tag(lam: float) -> str:
    """Filesystem-safe tag for a lambda value: 0.01 -> lam_0p01, 1.0 -> lam_1."""
    return "lam_" + f"{lam:g}".replace(".", "p")


# ------------------------------------------------------------------------- #
# core relabeling
# ------------------------------------------------------------------------- #


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


# ------------------------------------------------------------------------- #
# Pareto-point bookkeeping
# ------------------------------------------------------------------------- #


class PointAccumulator:
    """Streaming accumulator for (sum_ppl, sum_compute, count) per key."""

    def __init__(self) -> None:
        self.sum_ppl: Dict[str, float] = defaultdict(float)
        self.sum_cmp: Dict[str, float] = defaultdict(float)
        self.n: Dict[str, int] = defaultdict(int)

    def add(
        self,
        key: str,
        per_l_ppl: torch.Tensor,
        label_idx: torch.Tensor,
        l_inv: torch.Tensor,
    ) -> None:
        # per_l_ppl: [N, K] ; label_idx: [N] ; l_inv: [K]
        n = per_l_ppl.shape[0]
        idx_long = label_idx.to(torch.long)
        # gather PPL achieved at each row's chosen L
        ppl_per_row = per_l_ppl.gather(1, idx_long.unsqueeze(-1)).squeeze(-1)
        cmp_per_row = l_inv[idx_long]
        # filter out infinities (boundaries where _block_ppl_at returned inf)
        finite_mask = torch.isfinite(ppl_per_row)
        if finite_mask.any():
            self.sum_ppl[key] += float(ppl_per_row[finite_mask].sum().item())
            self.sum_cmp[key] += float(cmp_per_row[finite_mask].sum().item())
            self.n[key] += int(finite_mask.sum().item())

    def add_fixed(
        self,
        key: str,
        per_l_ppl: torch.Tensor,
        L_idx: int,
        l_inv: torch.Tensor,
    ) -> None:
        ppl_col = per_l_ppl[:, L_idx]
        finite_mask = torch.isfinite(ppl_col)
        if finite_mask.any():
            self.sum_ppl[key] += float(ppl_col[finite_mask].sum().item())
            self.sum_cmp[key] += float(l_inv[L_idx].item()) * int(finite_mask.sum().item())
            self.n[key] += int(finite_mask.sum().item())

    def mean(self, key: str) -> Tuple[float, float, int]:
        n = self.n.get(key, 0)
        if n == 0:
            return float("nan"), float("nan"), 0
        return self.sum_ppl[key] / n, self.sum_cmp[key] / n, n


# ------------------------------------------------------------------------- #
# AdaBlock reference: load matching teacher shards and align to oracle rows
# ------------------------------------------------------------------------- #


def _flatten_teacher_labels(
    teacher_shards: List[str],
) -> Optional[Dict[int, List[int]]]:
    """Return dict: prompt_id -> list of (AdaBlock label idx) in row order.

    The teacher and oracle pipelines both iterate prompts via iter_prompts()
    and record one row per boundary on the default-B=16 grid in order, so we
    can rebuild a per-prompt sequence by concatenating shards in sorted order.
    """
    if not teacher_shards:
        return None
    seq: Dict[int, List[int]] = defaultdict(list)
    for sp in sorted(teacher_shards):
        try:
            d = torch.load(sp, map_location="cpu")
        except Exception as e:
            print(f"[adablock-ref] could not load {sp}: {e}")
            continue
        labels = d.get("labels")
        prompt_ids = d.get("prompt_ids")
        if labels is None or prompt_ids is None:
            continue
        for pid, lab in zip(prompt_ids.tolist(), labels.tolist()):
            seq[int(pid)].append(int(lab))
    return seq


def _align_adablock_to_oracle(
    oracle_prompt_ids: torch.Tensor,
    teacher_seq: Dict[int, List[int]],
    cursor: Dict[int, int],
) -> Optional[torch.Tensor]:
    """For an oracle shard's prompt_ids in row order, return the AdaBlock
    labels for those same boundaries by advancing a per-prompt cursor through
    teacher_seq. Returns None if alignment fails for any row."""
    out = torch.zeros(oracle_prompt_ids.shape[0], dtype=torch.long)
    for i, pid in enumerate(oracle_prompt_ids.tolist()):
        pid = int(pid)
        seq = teacher_seq.get(pid)
        c = cursor.get(pid, 0)
        if seq is None or c >= len(seq):
            return None  # mismatch -> skip AdaBlock comparison entirely
        out[i] = int(seq[c])
        cursor[pid] = c + 1
    return out


# ------------------------------------------------------------------------- #
# verdict
# ------------------------------------------------------------------------- #


def _pareto_dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """a = (ppl, compute). a dominates b iff a is no worse on both and
    strictly better on at least one (lower is better for both)."""
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def render_verdict(
    model: str,
    lambdas: List[float],
    lam_pts: Dict[float, Tuple[float, float]],
    fixed_pts: Dict[int, Tuple[float, float]],
    adablock_pt: Optional[Tuple[float, float]],
) -> None:
    print(f"\n=== verdict for model={model} ===")

    # 1. all points table
    print("  candidates (lower PPL and lower compute is better):")
    print(f"    {'option':<20} {'achievable_PPL':>15} {'achievable_1/L':>16}")
    rows: List[Tuple[str, float, float]] = []
    for lam in lambdas:
        if lam in lam_pts:
            p = lam_pts[lam]
            rows.append((f"oracle  lam={lam:g}", p[0], p[1]))
    for B in CANDIDATE_BLOCK_SIZES:
        if B in fixed_pts:
            p = fixed_pts[B]
            rows.append((f"fixed-B={B}", p[0], p[1]))
    if adablock_pt is not None:
        rows.append(("adablock-rule", adablock_pt[0], adablock_pt[1]))
    for name, ppl, cmp in rows:
        print(f"    {name:<20} {ppl:>15.4f} {cmp:>16.5f}")

    # 2. lambda-vs-AdaBlock dominance
    if adablock_pt is None:
        print("  no AdaBlock reference available (teacher tree not found).")
    else:
        dominators = [
            lam for lam in lambdas
            if lam in lam_pts and _pareto_dominates(lam_pts[lam], adablock_pt)
        ]
        if dominators:
            best = min(
                dominators,
                key=lambda lam: (
                    (lam_pts[lam][0] - adablock_pt[0])
                    + (lam_pts[lam][1] - adablock_pt[1])
                ),
            )
            ppl, cmp = lam_pts[best]
            d_ppl = (adablock_pt[0] - ppl) / max(adablock_pt[0], 1e-9) * 100
            d_cmp = (adablock_pt[1] - cmp) / max(adablock_pt[1], 1e-9) * 100
            print(
                f"  Pareto-dominates AdaBlock: {dominators}  "
                f"(strictly lower PPL AND lower compute)"
            )
            print(
                f"  RECOMMENDED lambda (largest dominance margin): "
                f"lam={best:g}  "
                f"PPL  {d_ppl:+.1f}% vs AdaBlock,  "
                f"compute  {d_cmp:+.1f}% vs AdaBlock"
            )
        else:
            # closest to AdaBlock with lower PPL, fallback to lower compute
            quality_winners = [
                lam for lam in lambdas
                if lam in lam_pts and lam_pts[lam][0] < adablock_pt[0]
            ]
            speed_winners = [
                lam for lam in lambdas
                if lam in lam_pts and lam_pts[lam][1] < adablock_pt[1]
            ]
            print("  No lambda Pareto-dominates AdaBlock. Tradeoff picks:")
            if quality_winners:
                lam_q = min(quality_winners, key=lambda l: lam_pts[l][0])
                p, c = lam_pts[lam_q]
                print(
                    f"    Best-quality:   lam={lam_q:g}  "
                    f"PPL={p:.4f} (vs AdaBlock {adablock_pt[0]:.4f}), "
                    f"compute={c:.5f} (vs AdaBlock {adablock_pt[1]:.5f})"
                )
            else:
                print("    Best-quality:   no lambda beats AdaBlock on PPL.")
            if speed_winners:
                lam_s = min(speed_winners, key=lambda l: lam_pts[l][1])
                p, c = lam_pts[lam_s]
                print(
                    f"    Best-speed:     lam={lam_s:g}  "
                    f"PPL={p:.4f} (vs AdaBlock {adablock_pt[0]:.4f}), "
                    f"compute={c:.5f} (vs AdaBlock {adablock_pt[1]:.5f})"
                )
            else:
                print("    Best-speed:     no lambda beats AdaBlock on compute.")


# ------------------------------------------------------------------------- #
# main
# ------------------------------------------------------------------------- #


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

    l_inv = torch.tensor(
        [1.0 / L for L in CANDIDATE_BLOCK_SIZES], dtype=torch.float32
    )

    # Group oracle shards by (model, benchmark) so we can pre-load matching
    # teacher trees per model.
    by_mb: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for sp in shard_paths:
        rel = os.path.relpath(sp, os.path.join(args.in_dir, args.source))
        parts = rel.split(os.sep)
        if len(parts) < 3:
            continue
        by_mb[(parts[0], parts[1])].append(sp)

    # Optional: load matching AdaBlock (teacher) sequences per (model, benchmark).
    teacher_seq_by_mb: Dict[Tuple[str, str], Optional[Dict[int, List[int]]]] = {}
    if args.teacher_source:
        for (model, benchmark) in by_mb.keys():
            tdir = os.path.join(
                args.in_dir, args.teacher_source, model, benchmark
            )
            tshards = sorted(glob.glob(
                os.path.join(tdir, f"{args.teacher_prefix}_*.pt")
            ))
            teacher_seq_by_mb[(model, benchmark)] = (
                _flatten_teacher_labels(tshards) if tshards else None
            )

    # Accumulators
    relabel_hist: Dict[str, Counter] = {}  # f"{lam}|{model}|{benchmark}" -> Counter
    lam_acc = PointAccumulator()           # f"{lam}|{model}" or "{lam}|{model}|{bm}"
    fixed_acc = PointAccumulator()         # f"{model}|{B}"
    ada_acc = PointAccumulator()           # f"{model}"

    bytes_written = 0
    n_examples_total = 0
    t0 = time.time()
    n_processed = 0

    for (model, benchmark), oracle_paths in by_mb.items():
        teacher_seq = teacher_seq_by_mb.get((model, benchmark))
        ada_cursor: Dict[int, int] = {}
        ada_alignment_ok = teacher_seq is not None

        for sp in sorted(oracle_paths):
            try:
                shard = torch.load(sp, map_location="cpu")
            except Exception as e:
                print(f"[relabel] failed to load {sp}: {e}")
                continue

            if "per_l_ppl" not in shard:
                print(f"[relabel] skipping {sp} (no per_l_ppl)")
                continue

            per_l_ppl = shard["per_l_ppl"].to(torch.float32)
            prompt_ids = shard["prompt_ids"]
            n_examples_total += per_l_ppl.shape[0]

            # fixed-B reference points (from this oracle shard's per_l_ppl)
            for k, B in enumerate(CANDIDATE_BLOCK_SIZES):
                fixed_acc.add_fixed(f"{model}|{B}", per_l_ppl, k, l_inv)

            # AdaBlock reference (only if teacher tree aligned for all rows)
            if ada_alignment_ok:
                ada_labels = _align_adablock_to_oracle(
                    prompt_ids, teacher_seq, ada_cursor
                )
                if ada_labels is None:
                    ada_alignment_ok = False
                    print(
                        f"[adablock-ref] alignment lost on {model}/{benchmark} "
                        f"at shard {os.path.basename(sp)}; skipping for this model"
                    )
                else:
                    ada_acc.add(f"{model}", per_l_ppl, ada_labels, l_inv)

            # Per-lambda relabel + Pareto bookkeeping
            fname = os.path.basename(sp)
            for lam in lambdas:
                new_labels = relabel_per_l_ppl(per_l_ppl, lam)
                hist_key = f"{lam:g}|{model}|{benchmark}"
                hist = relabel_hist.setdefault(hist_key, Counter())
                for v in new_labels.tolist():
                    hist[int(v)] += 1
                lam_acc.add(f"{lam:g}|{model}", per_l_ppl, new_labels, l_inv)

                if args.dry_run:
                    continue

                out_subdir = os.path.join(
                    args.out_dir, _lam_tag(lam), args.source, model, benchmark
                )
                os.makedirs(out_subdir, exist_ok=True)
                out_path = os.path.join(out_subdir, fname)

                new_shard = dict(shard)
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

            n_processed += 1
            if n_processed % 20 == 0 or n_processed == len(shard_paths):
                elapsed = time.time() - t0
                rate = n_processed / max(elapsed, 1e-6)
                print(
                    f"[relabel] {n_processed}/{len(shard_paths)} shards processed "
                    f"({rate:.1f} shards/s, {elapsed:.1f}s elapsed)"
                )

    elapsed = time.time() - t0
    print(
        f"\n[relabel] done in {elapsed:.1f}s. "
        f"shards={len(shard_paths)} lambdas={len(lambdas)} "
        f"examples_per_lambda={n_examples_total} "
        f"bytes_written={bytes_written/1e9:.2f}GB"
    )

    # Per-(lambda x model x benchmark) class distribution
    print("\n=== relabel summary (label distribution per lambda x model x benchmark) ===")
    for lam in lambdas:
        print(f"\n  lambda = {lam}")
        rows = sorted(k for k in relabel_hist if k.startswith(f"{lam:g}|"))
        for key in rows:
            _, model, benchmark = key.split("|")
            c = relabel_hist[key]
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

    # Per-model verdict
    models = sorted({m for (m, _b) in by_mb.keys()})
    for model in models:
        lam_pts: Dict[float, Tuple[float, float]] = {}
        for lam in lambdas:
            ppl, cmp, n = lam_acc.mean(f"{lam:g}|{model}")
            if n > 0:
                lam_pts[lam] = (ppl, cmp)
        fixed_pts: Dict[int, Tuple[float, float]] = {}
        for B in CANDIDATE_BLOCK_SIZES:
            ppl, cmp, n = fixed_acc.mean(f"{model}|{B}")
            if n > 0:
                fixed_pts[B] = (ppl, cmp)
        ada_ppl, ada_cmp, ada_n = ada_acc.mean(f"{model}")
        ada_pt = (ada_ppl, ada_cmp) if ada_n > 0 else None

        render_verdict(model, lambdas, lam_pts, fixed_pts, ada_pt)


if __name__ == "__main__":
    main()
