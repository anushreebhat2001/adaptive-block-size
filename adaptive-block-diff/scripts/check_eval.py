"""Inspect evaluation JSONs from 04_eval_benchmarks.sbatch.

Auto-discovers every `<results_dir>/<model>_<benchmark>_<scheduler>[_<tag>].json`,
parses accuracy / tokens_per_second / block_size_histogram, and prints:

  1. A per-benchmark table comparing all schedulers on accuracy + throughput.
  2. A cross-benchmark headline table per scheduler.
  3. A Pareto frontier across (compute_cost, accuracy).
  4. A verdict: best scheduler per benchmark, biggest deployment win vs adablock.

Usage:
    python -m scripts.check_eval
    python -m scripts.check_eval --results_dir /scratch/$USER/Efficient-AI/results/base
    python -m scripts.check_eval --model llada
    python -m scripts.check_eval --benchmark gsm8k
    python -m scripts.check_eval --tag lam_10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple


CANDIDATE_BLOCK_SIZES = [4, 8, 16, 32]


def parse_args() -> argparse.Namespace:
    user = os.environ.get("USER", "")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results_dir",
        default=f"/scratch/{user}/Efficient-AI/results/base",
        help="Dir containing <model>_<bench>_<sched>[_<tag>].json files.",
    )
    p.add_argument("--model", default=None, help="Filter to one model (llada/dream).")
    p.add_argument("--benchmark", default=None, help="Filter to one benchmark.")
    p.add_argument("--scheduler", default=None, help="Filter to one scheduler.")
    p.add_argument(
        "--tag",
        default=None,
        help='Filter to one EVAL_TAG. "" matches base (no tag); '
             '"lam_10" matches *_lam_10.json. Default: show all.',
    )
    p.add_argument(
        "--show_per_prompt",
        action="store_true",
        help="Also print per-prompt scoring breakdown (verbose).",
    )
    return p.parse_args()


# ---------------------------------------------------- filename parsing ----


# Schedulers must be matched longest-first so "ours-teacher" is found before
# the substring "teacher" matches a tag containing teacher (it doesn't, but
# defensive sorting keeps this robust).
KNOWN_SCHEDULERS = sorted(
    [
        "fixed-4", "fixed-8", "fixed-16", "fixed-32",
        "adablock", "ours-teacher", "ours-oracle",
    ],
    key=len,
    reverse=True,
)


def parse_filename(path: str) -> Optional[Tuple[str, str, str, str]]:
    """Return (model, benchmark, scheduler, tag) or None if unparseable.

    Filename format: <model>_<benchmark>_<scheduler>[_<tag>].json
    Schedulers can contain hyphens (fixed-4, ours-teacher), so we look for a
    known scheduler token within the underscore-separated parts.
    """
    name = os.path.basename(path).replace(".json", "")
    parts = name.split("_")
    if len(parts) < 3:
        return None
    # Find the scheduler token by scanning. The scheduler is itself a single
    # underscore-free token (uses hyphens), so it lives at exactly one index.
    sched_idx = None
    for i, tok in enumerate(parts):
        if tok in KNOWN_SCHEDULERS:
            sched_idx = i
            break
    if sched_idx is None or sched_idx < 2:
        return None
    model = parts[0]
    benchmark = "_".join(parts[1:sched_idx])
    scheduler = parts[sched_idx]
    tag = "_".join(parts[sched_idx + 1 :])  # "" for base run
    return (model, benchmark, scheduler, tag)


def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def discover_evals(results_dir: str) -> List[Dict]:
    """Return list of eval records, one per JSON file."""
    out: List[Dict] = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        parsed = parse_filename(path)
        if parsed is None:
            continue
        model, benchmark, scheduler, tag = parsed
        data = load_json(path)
        if data is None:
            continue
        out.append({
            "model": model,
            "benchmark": benchmark,
            "scheduler": scheduler,
            "tag": tag,
            "path": path,
            "data": data,
        })
    return out


# ---------------------------------------------------------- metrics ------


def hist_compute(hist: Dict) -> float:
    """E[1/L] under a block-size histogram. Lower = larger blocks = cheaper."""
    if not hist:
        return float("nan")
    total = sum(hist.values())
    if total == 0:
        return float("nan")
    return sum(int(c) * (1.0 / int(L)) for L, c in hist.items()) / total


def hist_l1(h1: Dict, h2: Dict) -> float:
    """L1 distance between two block-size histograms (normalized)."""
    keys = set(h1) | set(h2)
    s1 = sum(h1.values()) or 1
    s2 = sum(h2.values()) or 1
    return sum(abs(int(h1.get(k, 0)) / s1 - int(h2.get(k, 0)) / s2) for k in keys)


def fmt_hist(hist: Dict) -> str:
    if not hist:
        return "(empty)"
    items = sorted(hist.items(), key=lambda kv: int(kv[0]))
    total = sum(hist.values()) or 1
    return " ".join(f"B{k}:{100 * int(v) / total:.0f}%" for k, v in items)


def get_acc(data: dict) -> float:
    for k in ("accuracy", "acc"):
        if k in data:
            return float(data[k])
    return float("nan")


def get_tps(data: dict) -> float:
    for k in ("tokens_per_second", "tps"):
        if k in data:
            return float(data[k])
    return float("nan")


def get_hist(data: dict) -> Dict:
    return data.get("block_size_histogram") or data.get("hist") or {}


# ---------------------------------------------------- printers ----------


def per_benchmark_table(records: List[Dict]) -> None:
    """Group by (model, benchmark) and show schedulers side by side."""
    by_mb: Dict[Tuple[str, str], List[Dict]] = {}
    for r in records:
        by_mb.setdefault((r["model"], r["benchmark"]), []).append(r)

    for (model, benchmark), rows in sorted(by_mb.items()):
        print("\n" + "=" * 100)
        print(f"{model} / {benchmark}")
        print("=" * 100)
        # Find adablock baseline (no-tag) for "vs_ada" computation.
        ada = None
        for r in rows:
            if r["scheduler"] == "adablock" and r["tag"] == "":
                ada = r
                break
        ada_hist = get_hist(ada["data"]) if ada else {}
        ada_acc = get_acc(ada["data"]) if ada else float("nan")
        ada_tps = get_tps(ada["data"]) if ada else float("nan")

        print(
            f"  {'scheduler':<14} {'tag':<10} {'acc':>6} {'t/s':>7} "
            f"{'cmp':>7} {'Δacc':>7} {'Δt/s':>7} {'vs_ada':>7}  histogram"
        )
        print("  " + "-" * 96)
        # Sort: fixed-* in numeric order, then adablock, then ours-*.
        order = {
            "fixed-4": 0, "fixed-8": 1, "fixed-16": 2, "fixed-32": 3,
            "adablock": 4, "ours-teacher": 5, "ours-oracle": 6,
        }
        rows_sorted = sorted(rows, key=lambda r: (order.get(r["scheduler"], 99), r["tag"]))
        for r in rows_sorted:
            d = r["data"]
            acc = get_acc(d)
            tps = get_tps(d)
            hist = get_hist(d)
            cmp_cost = hist_compute(hist)
            d_acc = acc - ada_acc if ada else float("nan")
            d_tps = tps - ada_tps if ada else float("nan")
            vs_ada = hist_l1(hist, ada_hist) if ada else 0.0
            tag_str = r["tag"] if r["tag"] else "-"
            print(
                f"  {r['scheduler']:<14} {tag_str:<10} "
                f"{acc:>6.3f} {tps:>7.1f} {cmp_cost:>7.4f} "
                f"{d_acc:>+7.3f} {d_tps:>+7.1f} {vs_ada:>7.3f}  {fmt_hist(hist)}"
            )

        if ada is None:
            print("  (no adablock baseline found for this (model, benchmark) — Δ columns are NaN)")


def cross_benchmark_table(records: List[Dict]) -> None:
    """One row per (model, scheduler, tag); columns are benchmarks. Shows acc."""
    by_msc: Dict[Tuple[str, str, str], Dict[str, Dict]] = {}
    benchmarks = set()
    for r in records:
        key = (r["model"], r["scheduler"], r["tag"])
        by_msc.setdefault(key, {})[r["benchmark"]] = r["data"]
        benchmarks.add(r["benchmark"])

    bms = sorted(benchmarks)
    if not bms:
        return

    for metric_name, getter, fmt in [
        ("accuracy", get_acc, "{:>7.3f}"),
        ("tokens_per_second", get_tps, "{:>7.1f}"),
    ]:
        print("\n" + "=" * 100)
        print(f"CROSS-BENCHMARK ({metric_name}) per (model, scheduler, tag)")
        print("=" * 100)
        header = f"  {'model':<8} {'scheduler':<14} {'tag':<10}"
        for b in bms:
            header += f" {b:>9}"
        header += f" {'avg':>9}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        order = {
            "fixed-4": 0, "fixed-8": 1, "fixed-16": 2, "fixed-32": 3,
            "adablock": 4, "ours-teacher": 5, "ours-oracle": 6,
        }
        for key, by_b in sorted(
            by_msc.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 99), kv[0][2])
        ):
            model, scheduler, tag = key
            tag_str = tag if tag else "-"
            row = f"  {model:<8} {scheduler:<14} {tag_str:<10}"
            vals = []
            for b in bms:
                if b in by_b:
                    v = getter(by_b[b])
                    vals.append(v)
                    row += " " + fmt.format(v)
                else:
                    row += "       --"
            avg = sum(vals) / len(vals) if vals else float("nan")
            if vals:
                row += " " + fmt.format(avg)
            print(row)


def pareto_frontier(records: List[Dict]) -> None:
    """Per model: scatter (compute, acc) and identify Pareto-optimal points."""
    by_model: Dict[str, List[Dict]] = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)

    for model, rows in sorted(by_model.items()):
        print("\n" + "=" * 100)
        print(f"PARETO FRONTIER ({model}): higher acc + lower compute is better")
        print("=" * 100)
        # Aggregate per (scheduler, tag) by averaging across benchmarks.
        agg: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        for r in rows:
            d = r["data"]
            acc = get_acc(d)
            cmp_cost = hist_compute(get_hist(d))
            agg.setdefault((r["scheduler"], r["tag"]), []).append((acc, cmp_cost))
        points: List[Tuple[str, str, float, float]] = []
        for (sch, tag), vals in agg.items():
            accs = [v[0] for v in vals]
            cmps = [v[1] for v in vals if not (v[1] != v[1])]  # filter NaN
            mean_acc = sum(accs) / len(accs) if accs else float("nan")
            mean_cmp = sum(cmps) / len(cmps) if cmps else float("nan")
            points.append((sch, tag, mean_acc, mean_cmp))

        # Pareto: dominated if some other point has acc >= self AND cmp <= self
        # with at least one strict.
        def dominates(a, b):
            return (a[2] >= b[2] and a[3] <= b[3]) and (a[2] > b[2] or a[3] < b[3])

        on_frontier = []
        for p in points:
            if not any(dominates(q, p) for q in points if q != p):
                on_frontier.append(p)

        # Print sorted by compute (cheapest first).
        print(f"  {'scheduler':<14} {'tag':<10} {'mean_acc':>9} {'mean_cmp':>9}  pareto?")
        print("  " + "-" * 60)
        for sch, tag, acc, cmp_cost in sorted(points, key=lambda x: x[3]):
            tag_str = tag if tag else "-"
            on = "✓" if (sch, tag, acc, cmp_cost) in on_frontier else " "
            print(
                f"  {sch:<14} {tag_str:<10} {acc:>9.3f} {cmp_cost:>9.4f}     {on}"
            )


def verdict(records: List[Dict]) -> None:
    """Per (model, benchmark): which scheduler best beats adablock on acc?"""
    print("\n" + "=" * 100)
    print("VERDICT (per model+benchmark: biggest accuracy win over adablock)")
    print("=" * 100)
    by_mb: Dict[Tuple[str, str], List[Dict]] = {}
    for r in records:
        by_mb.setdefault((r["model"], r["benchmark"]), []).append(r)

    for (model, benchmark), rows in sorted(by_mb.items()):
        ada = next((r for r in rows if r["scheduler"] == "adablock" and r["tag"] == ""), None)
        if ada is None:
            print(f"  {model}/{benchmark}: no adablock baseline available")
            continue
        ada_acc = get_acc(ada["data"])
        ada_tps = get_tps(ada["data"])
        # Best (acc, t/s) Pareto: highest acc; tiebreak by t/s.
        ours = [r for r in rows if r["scheduler"].startswith("ours-")]
        if not ours:
            print(f"  {model}/{benchmark}: no ours-* schedulers found")
            continue
        best = max(ours, key=lambda r: (get_acc(r["data"]), get_tps(r["data"])))
        b_acc = get_acc(best["data"])
        b_tps = get_tps(best["data"])
        d_acc = b_acc - ada_acc
        d_tps = b_tps - ada_tps
        tag_str = best["tag"] if best["tag"] else "-"
        verdict_str = (
            "WINS on acc" if d_acc > 0.005 else
            "TIES on acc" if abs(d_acc) <= 0.005 else
            "LOSES on acc"
        )
        if d_tps > 0:
            verdict_str += f", faster by {d_tps:+.1f} t/s"
        else:
            verdict_str += f", slower by {d_tps:+.1f} t/s"
        print(
            f"  {model:<8} {benchmark:<12} : best ours = {best['scheduler']:<13} tag={tag_str:<8}  "
            f"acc={b_acc:.3f} (Δ={d_acc:+.3f})  t/s={b_tps:.1f} (Δ={d_tps:+.1f})  -> {verdict_str}"
        )


# ---------------------------------------------------------- main --------


def main() -> None:
    args = parse_args()
    print(f"results_dir = {args.results_dir}")

    records = discover_evals(args.results_dir)
    if not records:
        print(f"\nno eval JSONs found under {args.results_dir}")
        return

    n_before = len(records)
    if args.model is not None:
        records = [r for r in records if r["model"] == args.model]
    if args.benchmark is not None:
        records = [r for r in records if r["benchmark"] == args.benchmark]
    if args.scheduler is not None:
        records = [r for r in records if r["scheduler"] == args.scheduler]
    if args.tag is not None:
        records = [r for r in records if r["tag"] == args.tag]

    if not records:
        print(f"\nno records match the filter; {n_before} discovered before filtering.")
        return

    print(
        f"discovered {n_before} eval JSONs"
        + (f"; {len(records)} after filter" if len(records) < n_before else "")
    )

    per_benchmark_table(records)
    cross_benchmark_table(records)
    pareto_frontier(records)
    verdict(records)


if __name__ == "__main__":
    main()
