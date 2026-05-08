"""Parse figures/eval_benchmarks.txt and emit headline figures into
figures/eval_plots/.

The txt file is the structured output of compare_smokes / lambda_sweep style
analysis: per (model, benchmark) tables with accuracy, tokens/sec, compute,
deltas vs adablock, and block-size histograms.

Plots produced:
  llada_accuracy_per_benchmark.png   — grouped bars: scheduler vs accuracy, one panel per benchmark
  llada_pareto_acc_vs_speed.png      — 2x2 acc-vs-t/s scatter (the "we beat X" plot)
  delta_vs_adablock_acc.png          — Δacc vs adablock per (benchmark, scheduler) (LLaDA)
  delta_vs_adablock_speed.png        — Δt/s vs adablock per (benchmark, scheduler) (LLaDA)
  headline_llada_humaneval.png       — the single biggest win, callout chart
  block_size_distribution_llada.png  — what each scheduler actually picks
  cross_benchmark_accuracy_llada.png — heatmap of accuracies across benchmarks
  speed_per_benchmark.png            — tokens/sec for both models across schedulers
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/utx/Desktop/code/adaptive-block-size")
TXT_PATH = ROOT / "figures" / "eval_benchmarks.txt"
OUT_DIR = ROOT / "figures" / "eval_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------- parser --

ROW_RE = re.compile(
    r"^\s*(?P<sched>[\w\-]+)\s+"
    r"(?P<tag>[\w\-]+)\s+"
    r"(?P<acc>-?\d+\.\d+)\s+"
    r"(?P<tps>-?\d+\.\d+)\s+"
    r"(?P<cmp>-?\d+\.\d+)\s+"
    r"(?P<dacc>[+-]?(?:nan|\d+\.\d+))\s+"
    r"(?P<dtps>[+-]?(?:nan|\d+\.\d+))\s+"
    r"(?P<vsada>-?\d+\.\d+)\s+"
    r"(?P<hist>.*)$"
)

HIST_PAIR = re.compile(r"B(\d+):(\d+)%")


def parse_histogram(s: str) -> Dict[int, int]:
    return {int(b): int(p) for b, p in HIST_PAIR.findall(s)}


def parse_txt(text: str) -> Dict:
    """Return per (model, benchmark) → list of row dicts, plus cross tables."""
    rows: List[Dict] = []
    section_title: Optional[str] = None
    in_table = False
    pending_model_bench: Optional[Tuple[str, str]] = None

    cross_acc: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    cross_tps: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    cross_mode: Optional[str] = None  # "acc" or "tps"

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("=" * 50):
            # next non-empty line is the title
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            title = lines[j].strip() if j < len(lines) else ""
            section_title = title
            # consume title and following === line
            i = j + 1
            while i < len(lines) and lines[i].startswith("=" * 50):
                i += 1
            in_table = True

            m = re.match(r"^([a-zA-Z]+)\s*/\s*([a-zA-Z0-9]+)$", title)
            if m:
                pending_model_bench = (m.group(1).lower(), m.group(2).lower())
                cross_mode = None
            elif "CROSS-BENCHMARK (accuracy)" in title:
                cross_mode = "acc"
                pending_model_bench = None
            elif "CROSS-BENCHMARK (tokens_per_second)" in title:
                cross_mode = "tps"
                pending_model_bench = None
            else:
                pending_model_bench = None
                cross_mode = None
            continue

        if in_table:
            if pending_model_bench is not None:
                m = ROW_RE.match(line)
                if m:
                    model, bench = pending_model_bench
                    d = m.groupdict()
                    rows.append({
                        "model": model,
                        "benchmark": bench,
                        "scheduler": d["sched"],
                        "tag": d["tag"] if d["tag"] != "-" else "",
                        "accuracy": float(d["acc"]),
                        "tokens_per_second": float(d["tps"]),
                        "compute": float(d["cmp"]),
                        "delta_acc": float("nan") if "nan" in d["dacc"] else float(d["dacc"]),
                        "delta_tps": float("nan") if "nan" in d["dtps"] else float(d["dtps"]),
                        "vs_ada": float(d["vsada"]),
                        "histogram": parse_histogram(d["hist"]),
                    })
            elif cross_mode is not None:
                # row format: model    scheduler      tag            gsm8k humaneval math mbpp avg
                m = re.match(
                    r"^\s*(\w+)\s+([\w\-]+)\s+([\w\-]+)\s+"
                    r"([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)",
                    line,
                )
                if m:
                    model, sched, tag = m.group(1), m.group(2), m.group(3)
                    if tag == "-":
                        tag = ""
                    vals = [m.group(i) for i in (4, 5, 6, 7, 8)]
                    headers = ["gsm8k", "humaneval", "math", "mbpp", "avg"]
                    parsed = {}
                    for h, v in zip(headers, vals):
                        try:
                            parsed[h] = float(v)
                        except ValueError:
                            parsed[h] = float("nan")
                    key = (model.lower(), sched, tag)
                    if cross_mode == "acc":
                        cross_acc[key] = parsed
                    else:
                        cross_tps[key] = parsed
        i += 1

    return {"rows": rows, "cross_acc": cross_acc, "cross_tps": cross_tps}


# -------------------------------------------------------------------- styling --

SCHEDULER_ORDER = [
    "fixed-4", "fixed-8", "fixed-16", "fixed-32",
    "adablock",
    "ours-teacher",
    "ours-oracle",
    "ours-oracle:lam_15",
]

COLOR = {
    "fixed-4":              "#a8a8a8",
    "fixed-8":              "#888888",
    "fixed-16":             "#666666",
    "fixed-32":             "#444444",
    "adablock":             "#ff7f0e",
    "ours-teacher":         "#1f77b4",
    "ours-oracle":          "#d62728",
    "ours-oracle:lam_15":   "#9467bd",
}


def label_of(row: Dict) -> str:
    if row["tag"]:
        return f"{row['scheduler']}:{row['tag']}"
    return row["scheduler"]


# ---------------------------------------------------------------------- plots --

BENCHMARKS = ["gsm8k", "humaneval", "math", "mbpp"]


def fig_llada_accuracy_per_benchmark(rows: List[Dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    for ax, bench in zip(axes, BENCHMARKS):
        bench_rows = [r for r in rows if r["model"] == "llada" and r["benchmark"] == bench]
        bench_rows.sort(key=lambda r: SCHEDULER_ORDER.index(label_of(r))
                        if label_of(r) in SCHEDULER_ORDER else 99)
        labels = [label_of(r) for r in bench_rows]
        accs = [r["accuracy"] for r in bench_rows]
        colors = [COLOR.get(l, "#999999") for l in labels]
        bars = ax.bar(range(len(labels)), accs, color=colors, edgecolor="black", linewidth=0.5)
        # Highlight best ours-* with a hatch
        ours_idx = [i for i, l in enumerate(labels) if l.startswith("ours")]
        ada_idx = [i for i, l in enumerate(labels) if l == "adablock"]
        if ada_idx and ours_idx:
            ada_acc = accs[ada_idx[0]]
            for i in ours_idx:
                if accs[i] > ada_acc:
                    bars[i].set_edgecolor("green")
                    bars[i].set_linewidth(2.0)
        for i, v in enumerate(accs):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("accuracy")
        ax.set_title(f"LLaDA / {bench}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, max(accs + [0.01]) * 1.18)
    fig.suptitle("LLaDA accuracy per benchmark — green outline = ours-* beats adablock",
                 fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / "llada_accuracy_per_benchmark.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_llada_pareto_acc_vs_speed(rows: List[Dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    for ax, bench in zip(axes, BENCHMARKS):
        bench_rows = [r for r in rows if r["model"] == "llada" and r["benchmark"] == bench]
        for r in bench_rows:
            l = label_of(r)
            color = COLOR.get(l, "#999999")
            marker = "o" if l.startswith("fixed") else ("s" if l.startswith("ours") else "D")
            size = 110 if l.startswith("ours") or l == "adablock" else 70
            ax.scatter(r["tokens_per_second"], r["accuracy"], color=color,
                       s=size, marker=marker, edgecolor="black", linewidth=0.6,
                       label=l, zorder=3)
            ax.annotate(l, (r["tokens_per_second"], r["accuracy"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
        # connect fixed-* into a line
        fixed = sorted(
            [r for r in bench_rows if r["scheduler"].startswith("fixed-")],
            key=lambda r: int(r["scheduler"].split("-")[1])
        )
        if fixed:
            xs = [r["tokens_per_second"] for r in fixed]
            ys = [r["accuracy"] for r in fixed]
            ax.plot(xs, ys, color="#888888", linestyle="--", linewidth=1, zorder=1)
        ax.set_xlabel("tokens / second  (faster →)")
        ax.set_ylabel("accuracy  (↑ better)")
        ax.set_title(f"LLaDA / {bench}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("LLaDA: accuracy vs throughput  (top-right is best)", fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / "llada_pareto_acc_vs_speed.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_delta_vs_adablock(rows: List[Dict], metric: str, ylabel: str, fname: str) -> None:
    """Bar chart of Δ-metric vs adablock per (benchmark, ours-* scheduler)."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    schedulers = ["ours-teacher", "ours-oracle", "ours-oracle:lam_15"]
    n_bench = len(BENCHMARKS)
    n_sched = len(schedulers)
    width = 0.22
    x = np.arange(n_bench)

    for i, sched in enumerate(schedulers):
        vals = []
        for bench in BENCHMARKS:
            match = [
                r for r in rows
                if r["model"] == "llada" and r["benchmark"] == bench and label_of(r) == sched
            ]
            if not match or np.isnan(match[0][metric]):
                vals.append(0.0)
            else:
                vals.append(match[0][metric])
        bars = ax.bar(x + (i - (n_sched - 1) / 2) * width, vals, width,
                      label=sched, color=COLOR[sched],
                      edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            if v == 0.0:
                continue
            ax.text(b.get_x() + b.get_width() / 2, v + (0.001 if metric == "delta_acc" else 0.05),
                    f"{v:+.3f}" if metric == "delta_acc" else f"{v:+.1f}",
                    ha="center",
                    va="bottom" if v > 0 else "top",
                    fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS)
    ax.set_ylabel(ylabel)
    ax.set_title(f"LLaDA: {ylabel} vs AdaBlock baseline (positive = ours wins)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_headline_llada_humaneval(rows: List[Dict]) -> None:
    bench_rows = [r for r in rows if r["model"] == "llada" and r["benchmark"] == "humaneval"]
    bench_rows.sort(key=lambda r: SCHEDULER_ORDER.index(label_of(r))
                    if label_of(r) in SCHEDULER_ORDER else 99)
    labels = [label_of(r) for r in bench_rows]
    accs = [r["accuracy"] for r in bench_rows]
    speeds = [r["tokens_per_second"] for r in bench_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = [COLOR.get(l, "#999999") for l in labels]
    bars = ax1.bar(range(len(labels)), accs, color=colors, edgecolor="black", linewidth=0.6)
    # callout the winner
    win_idx = labels.index("ours-teacher") if "ours-teacher" in labels else None
    if win_idx is not None:
        bars[win_idx].set_edgecolor("green")
        bars[win_idx].set_linewidth(3)
    for i, v in enumerate(accs):
        ax1.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("accuracy")
    ax1.set_title("Accuracy on LLaDA / HumanEval")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_ylim(0, max(accs) * 1.12)

    bars2 = ax2.bar(range(len(labels)), speeds, color=colors, edgecolor="black", linewidth=0.6)
    if win_idx is not None:
        bars2[win_idx].set_edgecolor("green")
        bars2[win_idx].set_linewidth(3)
    for i, v in enumerate(speeds):
        ax2.text(i, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("tokens / second")
    ax2.set_title("Throughput on LLaDA / HumanEval")
    ax2.grid(True, axis="y", alpha=0.3)

    if "adablock" in labels and "ours-teacher" in labels:
        ada_acc = accs[labels.index("adablock")]
        ours_acc = accs[labels.index("ours-teacher")]
        ada_t = speeds[labels.index("adablock")]
        ours_t = speeds[labels.index("ours-teacher")]
        fig.suptitle(
            f"Headline win — LLaDA / HumanEval:  "
            f"ours-teacher = {ours_acc:.3f}  vs  adablock = {ada_acc:.3f}  "
            f"(+{(ours_acc - ada_acc):.3f} acc,  +{(ours_t - ada_t):+.1f} t/s)",
            fontsize=13,
        )
    fig.tight_layout()
    out = OUT_DIR / "headline_llada_humaneval.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_block_size_distribution(rows: List[Dict]) -> None:
    """Stacked horizontal bars: % of blocks at each size, per scheduler.
    Pooled across LLaDA benchmarks."""
    SIZES = [4, 8, 12, 16, 20, 24, 28, 32]
    schedulers = ["fixed-4", "fixed-8", "fixed-16", "fixed-32", "adablock",
                  "ours-teacher", "ours-oracle", "ours-oracle:lam_15"]
    cmap = plt.colormaps.get_cmap("viridis")
    size_color = {b: cmap(i / (len(SIZES) - 1)) for i, b in enumerate(SIZES)}

    pooled: Dict[str, Dict[int, float]] = {}
    for sched in schedulers:
        rs = [r for r in rows if r["model"] == "llada" and label_of(r) == sched]
        if not rs:
            continue
        agg: Dict[int, float] = {b: 0.0 for b in SIZES}
        n = 0
        for r in rs:
            for b, p in r["histogram"].items():
                agg[b] = agg.get(b, 0) + p
            n += 1
        if n > 0:
            pooled[sched] = {b: agg[b] / n for b in SIZES}

    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(pooled))
    left = np.zeros(len(pooled))
    sched_keys = list(pooled.keys())
    for b in SIZES:
        vals = np.array([pooled[s].get(b, 0.0) for s in sched_keys])
        ax.barh(y, vals, left=left, color=size_color[b], label=f"B={b}",
                edgecolor="white", linewidth=0.4)
        for i, v in enumerate(vals):
            if v >= 6:
                ax.text(left[i] + v / 2, i, f"{int(round(v))}%",
                        ha="center", va="center", fontsize=8, color="white")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(sched_keys)
    ax.set_xlabel("% of blocks (pooled across LLaDA benchmarks)")
    ax.set_title("What block sizes does each scheduler actually pick? (LLaDA)")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", ncol=4, fontsize=8, framealpha=0.95)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "block_size_distribution_llada.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_cross_benchmark_acc(cross_acc: Dict) -> None:
    schedulers = ["fixed-4", "fixed-8", "fixed-16", "fixed-32", "adablock",
                  "ours-teacher", "ours-oracle", "ours-oracle:lam_15"]
    bench_cols = ["gsm8k", "humaneval", "math", "mbpp", "avg"]
    grid = np.full((len(schedulers), len(bench_cols)), np.nan)
    for i, sched in enumerate(schedulers):
        if ":" in sched:
            base, tag = sched.split(":", 1)
        else:
            base, tag = sched, ""
        key = ("llada", base, tag)
        if key not in cross_acc:
            continue
        for j, b in enumerate(bench_cols):
            grid[i, j] = cross_acc[key].get(b, np.nan)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black", fontsize=9)
            else:
                ax.text(j, i, "—", ha="center", va="center", color="black", fontsize=9)
    ax.set_xticks(range(len(bench_cols)))
    ax.set_xticklabels(bench_cols)
    ax.set_yticks(range(len(schedulers)))
    ax.set_yticklabels(schedulers)
    ax.set_title("LLaDA accuracy across benchmarks (— = no run)")
    fig.colorbar(im, ax=ax, label="accuracy")
    fig.tight_layout()
    out = OUT_DIR / "cross_benchmark_accuracy_llada.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_speed_per_benchmark(rows: List[Dict]) -> None:
    """Tokens/sec per scheduler per benchmark, both models side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)
    for ax, model in zip(axes, ["llada", "dream"]):
        x = np.arange(len(BENCHMARKS))
        schedulers = ["fixed-16", "adablock", "ours-teacher", "ours-oracle:lam_15"]
        width = 0.2
        for i, sched in enumerate(schedulers):
            vals = []
            for bench in BENCHMARKS:
                match = [
                    r for r in rows
                    if r["model"] == model and r["benchmark"] == bench and label_of(r) == sched
                ]
                vals.append(match[0]["tokens_per_second"] if match else 0.0)
            ax.bar(x + (i - (len(schedulers) - 1) / 2) * width, vals, width,
                   label=sched, color=COLOR[sched], edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(BENCHMARKS)
        ax.set_ylabel("tokens / second")
        ax.set_title(f"{model.upper()} throughput by benchmark")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Throughput per benchmark (selected schedulers)", fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / "speed_per_benchmark.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------- main --

def main() -> None:
    text = TXT_PATH.read_text()
    parsed = parse_txt(text)
    rows = parsed["rows"]
    cross_acc = parsed["cross_acc"]
    print(f"parsed {len(rows)} rows across {len({(r['model'], r['benchmark']) for r in rows})} (model, benchmark) cells")

    fig_llada_accuracy_per_benchmark(rows)
    fig_llada_pareto_acc_vs_speed(rows)
    fig_delta_vs_adablock(rows, "delta_acc", "Δ accuracy",
                          "delta_vs_adablock_acc.png")
    fig_delta_vs_adablock(rows, "delta_tps", "Δ tokens / second",
                          "delta_vs_adablock_speed.png")
    fig_headline_llada_humaneval(rows)
    fig_block_size_distribution(rows)
    fig_cross_benchmark_acc(cross_acc)
    fig_speed_per_benchmark(rows)


if __name__ == "__main__":
    main()
