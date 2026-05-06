"""Compare smoke-test results across the four model variants.

Reads from ${SMOKE_DIR}/{instruct,base,dream,dream_base}/ and prints a
side-by-side table for label quality (entropy, class balance), predictor
quality (val_acc, gain over majority, ECE), and deployment quality
(eval accuracy, tokens/sec, block-size histogram divergence).

Usage:
    python -m scripts.compare_smokes
    python -m scripts.compare_smokes --smoke_dir /scratch/$USER/Efficient-AI/smoke
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import torch


CANDS = [4, 8, 16, 32]


# ---------------------------------------------------------------- variants ---

# (display_name, subdir under smoke_dir, model name in shard paths)
VARIANTS = [
    ("LLaDA-Instruct", "instruct", "llada"),
    ("LLaDA-Base", "base", "llada"),
    ("Dream-Instruct", "dream", "dream"),
    ("Dream-Base", "dream_base", "dream"),
]


# ------------------------------------------------------------- label stats ---


def label_stats(shard_glob: str) -> Optional[Dict]:
    shards = sorted(glob.glob(shard_glob))
    if not shards:
        return None
    labels = torch.cat([torch.load(p, map_location="cpu")["labels"] for p in shards])
    counts = torch.bincount(labels, minlength=4).tolist()
    n = sum(counts)
    if n == 0:
        return None
    pcts = [100 * c / n for c in counts]
    probs = [c / n for c in counts if c > 0]
    H = -sum(p * math.log(p) for p in probs)
    nz = sum(1 for c in counts if c > 0)
    return {
        "n": n,
        "counts": counts,
        "pcts": pcts,
        "entropy": H,
        "nonzero_classes": nz,
        "max_pct": max(pcts),
    }


# --------------------------------------------------------- predictor stats ---


def predictor_stats(ckpt_path: str) -> Optional[Dict]:
    """Read the last entry of the per-epoch JSONL log next to the ckpt."""
    log = ckpt_path + ".log.jsonl"
    if not os.path.isfile(log):
        return None
    last_best = None  # last "saved new best" record
    last_any = None
    with open(log) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val_acc" not in d:
                continue
            last_any = d
            if d.get("is_best"):
                last_best = d
    return last_best or last_any


# -------------------------------------------------------------- eval stats ---


def eval_stats(json_path: str) -> Optional[Dict]:
    if not os.path.isfile(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)


def hist_diff(h1: Dict, h2: Dict) -> float:
    """L1 distance between two block-size histograms (normalized)."""
    keys = set(h1) | set(h2)
    s1 = sum(h1.values()) or 1
    s2 = sum(h2.values()) or 1
    return sum(abs(h1.get(k, 0) / s1 - h2.get(k, 0) / s2) for k in keys)


# --------------------------------------------------------------- printers ---


def print_label_table(rows: List[Tuple[str, Optional[Dict], Optional[Dict]]]) -> None:
    print("\n" + "=" * 90)
    print("LABEL QUALITY (gsm8k smoke shards)")
    print("=" * 90)
    print(f"{'variant':<18} {'src':<8} {'N':>6} {'cls':>5} {'H':>6} "
          f"{'B4%':>6} {'B8%':>6} {'B16%':>6} {'B32%':>6} {'maxPct':>8}")
    print("-" * 90)
    for name, t, o in rows:
        for src, s in [("teacher", t), ("oracle", o)]:
            if s is None:
                print(f"{name:<18} {src:<8} {'(missing)':>6}")
                continue
            print(
                f"{name:<18} {src:<8} {s['n']:>6} "
                f"{s['nonzero_classes']:>3}/4 {s['entropy']:>6.3f} "
                f"{s['pcts'][0]:>5.1f}% {s['pcts'][1]:>5.1f}% "
                f"{s['pcts'][2]:>5.1f}% {s['pcts'][3]:>5.1f}% "
                f"{s['max_pct']:>7.1f}%"
            )
    print()
    print("Healthy = cls 4/4, H > 1.0, maxPct < 70%")


def print_predictor_table(
    rows: List[Tuple[str, Optional[Dict], Optional[Dict]]]
) -> None:
    print("\n" + "=" * 90)
    print("PREDICTOR QUALITY (best-val-loss epoch)")
    print("=" * 90)
    print(f"{'variant':<18} {'src':<8} {'val_acc':>8} {'majority':>9} "
          f"{'gain':>8} {'val_off1':>9} {'ECE':>7}")
    print("-" * 90)
    for name, t, o in rows:
        for src, s in [("teacher", t), ("oracle", o)]:
            if s is None:
                print(f"{name:<18} {src:<8} {'(missing)':>8}")
                continue
            gain = s.get("gain_over_majority", float("nan"))
            ece = s.get("ece", float("nan"))
            off1 = s.get("val_off1", float("nan"))
            print(
                f"{name:<18} {src:<8} "
                f"{s['val_acc']:>8.3f} {s.get('majority', float('nan')):>9.3f} "
                f"{gain:>+8.3f} {off1:>9.3f} {ece:>7.3f}"
            )
    print()
    print("Headline: gain (val_acc - majority) is the metric controlling for "
          "label imbalance.")


def print_eval_table(
    rows: List[Tuple[str, Dict[str, Optional[Dict]]]]
) -> None:
    print("\n" + "=" * 90)
    print("DEPLOYMENT (eval on gsm8k test split)")
    print("=" * 90)
    print(f"{'variant':<18} {'scheduler':<14} {'acc':>6} {'t/s':>7} "
          f"{'hist':<32} {'vs_ada':>8}")
    print("-" * 90)
    for name, ev in rows:
        ada = ev.get("adablock")
        for sched in ("ours_teacher", "ours_oracle", "adablock"):
            d = ev.get(sched)
            if d is None:
                continue
            acc = d.get("accuracy", float("nan"))
            tps = d.get("tokens_per_second", float("nan"))
            hist = d.get("block_size_histogram", {})
            hist_str = " ".join(f"B{k}:{v}" for k, v in sorted(hist.items(), key=lambda kv: int(kv[0])))
            divergence = (
                hist_diff({int(k): v for k, v in hist.items()},
                          {int(k): v for k, v in ada["block_size_histogram"].items()})
                if ada and sched != "adablock"
                else 0.0
            )
            print(
                f"{name:<18} {sched:<14} {acc:>6.3f} {tps:>7.1f} "
                f"{hist_str:<32} {divergence:>+8.3f}"
            )
    print()
    print("vs_ada = L1 distance from adablock's block-size distribution. "
          "Higher = predictor's choices differ more from the baseline.")


# ---------------------------------------------------------------- ranking ---


def overall_ranking(
    rows_label: List[Tuple[str, Optional[Dict], Optional[Dict]]],
    rows_pred: List[Tuple[str, Optional[Dict], Optional[Dict]]],
) -> None:
    print("\n" + "=" * 90)
    print("RANKING (oracle-source signals only — that's the deployable predictor)")
    print("=" * 90)
    scores = []
    for (name, _, label_o), (_, _, pred_o) in zip(rows_label, rows_pred):
        if label_o is None or pred_o is None:
            scores.append((name, None, "incomplete"))
            continue
        # Higher is better on entropy, val_acc, gain. Lower is better on ECE.
        composite = (
            label_o["entropy"] / math.log(4)            # in [0, 1]
            + pred_o["val_acc"]                          # in [0, 1]
            + max(0.0, pred_o.get("gain_over_majority", 0.0)) * 2  # gain weighted 2x
            - pred_o.get("ece", 0.0)                     # lower better
        )
        scores.append((name, composite, "ok"))

    ranked = sorted(
        [s for s in scores if s[1] is not None],
        key=lambda x: -x[1],
    )
    for i, (name, score, _) in enumerate(ranked, 1):
        print(f"  {i}. {name:<18}  composite={score:.3f}")
    for name, _, status in scores:
        if status == "incomplete":
            print(f"  -- {name:<18}  (data missing — not ranked)")
    print()
    print("Composite = (entropy/ln4) + val_acc + 2*max(0,gain) - ECE.")
    print("Use this as a starting point; the eval table above is the real "
          "deployment signal once eval JSONs exist.")


# -------------------------------------------------------------------- main ---


def main() -> None:
    ap = argparse.ArgumentParser()
    user = os.environ.get("USER", "")
    ap.add_argument(
        "--smoke_dir",
        default=f"/scratch/{user}/Efficient-AI/smoke",
        help="Root smoke dir; expects subdirs instruct/, base/, dream/, dream_base/",
    )
    ap.add_argument("--benchmark", default="gsm8k")
    args = ap.parse_args()

    print(f"smoke_dir = {args.smoke_dir}")
    print(f"benchmark = {args.benchmark}")

    label_rows: List[Tuple[str, Optional[Dict], Optional[Dict]]] = []
    pred_rows: List[Tuple[str, Optional[Dict], Optional[Dict]]] = []
    eval_rows: List[Tuple[str, Dict[str, Optional[Dict]]]] = []

    for display, subdir, mname in VARIANTS:
        base = os.path.join(args.smoke_dir, subdir)
        # Labels
        t_glob = f"{base}/labels/{mname}/{args.benchmark}/*.pt"
        o_glob = f"{base}/oracle/labels/{mname}/{args.benchmark}/*.pt"
        label_rows.append((display, label_stats(t_glob), label_stats(o_glob)))
        # Predictors
        t_ckpt = f"{base}/ckpt/{mname}_teacher.pt"
        o_ckpt = f"{base}/oracle/ckpt/{mname}_oracle.pt"
        pred_rows.append((display, predictor_stats(t_ckpt), predictor_stats(o_ckpt)))
        # Eval
        ev = {
            "ours_teacher": eval_stats(f"{base}/results/ours_teacher.json"),
            "adablock":     eval_stats(f"{base}/results/adablock.json"),
            "ours_oracle":  eval_stats(f"{base}/oracle/results/ours_oracle.json"),
            # Adablock from oracle smoke if present (more recent eval config)
        }
        # Prefer oracle-smoke's adablock baseline if it exists
        ada_oracle = eval_stats(f"{base}/oracle/results/adablock.json")
        if ada_oracle:
            ev["adablock"] = ada_oracle
        eval_rows.append((display, ev))

    print_label_table(label_rows)
    print_predictor_table(pred_rows)
    print_eval_table(eval_rows)
    overall_ranking(label_rows, pred_rows)


if __name__ == "__main__":
    main()
