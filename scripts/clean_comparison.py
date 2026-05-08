"""Strict, prompt-aligned comparison: fixed-* vs ours-teacher vs ours-oracle.

Reads run_benchmarks.py output JSONs from a directory, then for each
(model, benchmark) builds a comparison restricted to the **intersection of
prompt_ids** that every scheduler-of-interest successfully completed. This
removes the unfairness that comes from one scheduler crashing on a different
set of prompts than another.

Outputs:
  - A printable report (per (model, benchmark) and a summary table).
  - A JSON dump of the cleaned per-cell metrics for plotting.
  - A markdown table you can paste into a writeup.

The goal is to be ruthless about apples-to-apples: same prompt set, same
scoring, same throughput definition. No averaging across crashed cells.

Usage:
    python scripts/clean_comparison.py \
        --results_dir results/base \
        --out_json    figures/clean_comparison.json \
        --out_md      figures/clean_comparison.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# Schedulers we treat as the "headline comparison set". Restricted set so the
# intersection-of-prompts isn't tiny when one scheduler had a partial run.
HEADLINE_SCHEDULERS = [
    "fixed-4", "fixed-8", "fixed-16", "fixed-32",
    "adablock",
    "ours-teacher",
    "ours-oracle",
]


def load_results(results_dir: Path) -> List[Dict]:
    """Read every *.json under results_dir."""
    out: List[Dict] = []
    for p in sorted(results_dir.rglob("*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[skip] {p}: {e}")
            continue
        if not all(k in data for k in ("model", "benchmark", "scheduler", "results")):
            continue
        out.append(data)
    return out


def cell_key(d: Dict) -> Tuple[str, str, str, str]:
    """Identify a result cell uniquely."""
    sched = d["scheduler"]
    # "ours-oracle" + tag (e.g. _lam_15) embedded in filename — capture from "lambda" if present
    tag = ""
    lam = d.get("lambda", None)
    if sched == "ours-oracle" and lam is not None and lam != 0.05:
        tag = f"lam_{int(round(lam * 100)):02d}"
    return (d["model"], d["benchmark"], sched, tag)


def per_prompt_table(d: Dict) -> Dict[int, Dict]:
    """prompt_id -> {correct, n_new_tokens, decode_seconds, block_sizes}."""
    return {int(r["prompt_id"]): r for r in d["results"]}


def intersect_prompts(cells: List[Dict]) -> Set[int]:
    """Set of prompt_ids that appear in every cell."""
    if not cells:
        return set()
    common = {int(r["prompt_id"]) for r in cells[0]["results"]}
    for c in cells[1:]:
        common &= {int(r["prompt_id"]) for r in c["results"]}
    return common


def metrics_on_subset(d: Dict, prompts: Set[int]) -> Dict:
    """Recompute accuracy and tokens/sec on the subset of prompts."""
    n_total = 0
    n_correct = 0
    total_tokens = 0
    total_seconds = 0.0
    hist: Dict[int, int] = {}
    for r in d["results"]:
        pid = int(r["prompt_id"])
        if pid not in prompts:
            continue
        n_total += 1
        n_correct += int(r["correct"])
        total_tokens += int(r["n_new_tokens"])
        total_seconds += float(r["decode_seconds"])
        for b in r.get("block_sizes", []):
            hist[int(b)] = hist.get(int(b), 0) + 1
    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": n_correct / max(n_total, 1),
        "tokens_per_second": total_tokens / max(total_seconds, 1e-9),
        "decode_seconds": total_seconds,
        "block_size_histogram": hist,
    }


def fmt_pct(v: float) -> str:
    return f"{v * 100:5.1f}%"


def fmt_num(v: float, w: int = 5, p: int = 1) -> str:
    return f"{v:{w}.{p}f}"


def render_table(model: str, benchmark: str, cells: Dict[Tuple[str, str], Dict],
                 baseline: Tuple[str, str]) -> str:
    """ASCII table comparing every (scheduler, tag) on a fixed prompt subset."""
    lines = []
    lines.append(f"\n=== {model} / {benchmark} ===")
    if baseline not in cells:
        lines.append(f"  (baseline {baseline} missing)")
        baseline = None  # type: ignore

    headers = ["scheduler", "tag", "n", "acc", "Δacc", "t/s", "Δt/s", "histogram"]
    rows = []
    base_acc = cells[baseline]["accuracy"] if baseline else None
    base_tps = cells[baseline]["tokens_per_second"] if baseline else None
    for (sched, tag), m in cells.items():
        hist_str = " ".join(
            f"B{b}:{int(round(100 * c / max(sum(m['block_size_histogram'].values()), 1)))}%"
            for b, c in sorted(m["block_size_histogram"].items())
        )
        d_acc = (m["accuracy"] - base_acc) if base_acc is not None else float("nan")
        d_tps = (m["tokens_per_second"] - base_tps) if base_tps is not None else float("nan")
        rows.append([
            sched,
            tag or "-",
            str(m["n_total"]),
            f"{m['accuracy']:.3f}",
            f"{d_acc:+.3f}" if not np.isnan(d_acc) else "  -  ",
            f"{m['tokens_per_second']:.1f}",
            f"{d_tps:+.1f}" if not np.isnan(d_tps) else "  -  ",
            hist_str,
        ])
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    lines.append(fmt.format(*headers))
    lines.append("  " + "-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        lines.append(fmt.format(*r))
    return "\n".join(lines)


def render_markdown(model: str, benchmark: str,
                    cells: Dict[Tuple[str, str], Dict],
                    baseline: Tuple[str, str]) -> str:
    """Markdown table for a writeup."""
    lines = [f"\n### {model} / {benchmark}\n"]
    base_acc = cells[baseline]["accuracy"] if baseline in cells else None
    base_tps = cells[baseline]["tokens_per_second"] if baseline in cells else None
    n_set = next(iter(cells.values()))["n_total"] if cells else 0
    lines.append(f"_n = {n_set} prompts (intersection across schedulers)_\n")
    lines.append("| scheduler | acc | Δacc vs adablock | t/s | Δt/s vs adablock |")
    lines.append("|---|---|---|---|---|")
    for (sched, tag), m in cells.items():
        label = sched if not tag else f"{sched}:{tag}"
        d_acc_str = f"{(m['accuracy'] - base_acc):+.3f}" if base_acc is not None else "—"
        d_tps_str = f"{(m['tokens_per_second'] - base_tps):+.1f}" if base_tps is not None else "—"
        lines.append(
            f"| **{label}** | {m['accuracy']:.3f} | {d_acc_str} | "
            f"{m['tokens_per_second']:.1f} | {d_tps_str} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True, type=Path)
    p.add_argument("--out_json", default=None, type=Path)
    p.add_argument("--out_md", default=None, type=Path)
    p.add_argument("--baseline", default="adablock",
                   help="scheduler used as the Δ-reference. Use 'fixed-16' to compare against the strongest fixed.")
    p.add_argument("--include_lambda_variants", action="store_true",
                   help="Also include ours-oracle:lam_* variants in the comparison.")
    args = p.parse_args()

    runs = load_results(args.results_dir)
    print(f"loaded {len(runs)} JSONs from {args.results_dir}")

    # Group by (model, benchmark)
    by_mb: Dict[Tuple[str, str], List[Dict]] = {}
    for r in runs:
        sched = r["scheduler"]
        if sched not in HEADLINE_SCHEDULERS:
            continue
        # Skip lambda variants unless requested
        lam = r.get("lambda", 0.05)
        if not args.include_lambda_variants and sched == "ours-oracle" and abs(lam - 0.05) > 1e-6:
            continue
        by_mb.setdefault((r["model"], r["benchmark"]), []).append(r)

    out_json: Dict = {}
    md_lines: List[str] = ["# Clean comparison report\n",
                           f"_baseline = `{args.baseline}`_\n"]
    for (model, bench), cells in sorted(by_mb.items()):
        if len(cells) < 2:
            print(f"[skip] {model}/{bench}: only {len(cells)} cells")
            continue
        common = intersect_prompts(cells)
        if len(common) == 0:
            print(f"[skip] {model}/{bench}: no common prompts across schedulers")
            continue
        # Recompute every cell on the common subset
        cell_metrics: Dict[Tuple[str, str], Dict] = {}
        for c in cells:
            tag = ""
            lam = c.get("lambda", 0.05)
            if c["scheduler"] == "ours-oracle" and abs(lam - 0.05) > 1e-6:
                tag = f"lam_{int(round(lam * 100)):02d}"
            cell_metrics[(c["scheduler"], tag)] = metrics_on_subset(c, common)

        baseline_key = (args.baseline, "")
        print(render_table(model, bench, cell_metrics, baseline_key))
        md_lines.append(render_markdown(model, bench, cell_metrics, baseline_key))
        out_json[f"{model}/{bench}"] = {
            "n_common_prompts": len(common),
            "cells": {f"{s}{':' + t if t else ''}": m for (s, t), m in cell_metrics.items()},
        }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out_json, f, indent=2)
        print(f"\nwrote {args.out_json}")

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text("\n".join(md_lines))
        print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
