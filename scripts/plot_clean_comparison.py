"""Single clean-comparison figure built from clean_comparison.json.

Reads the prompt-aligned per-cell metrics produced by clean_comparison.py and
emits one PNG per (model, benchmark) plus a combined headline figure.

Why a separate plot script: clean_comparison.json is recomputed on the
intersection-of-prompts subset, so n is identical across schedulers within a
panel. Plotting from this JSON guarantees the visual matches the report.

Usage:
    python scripts/plot_clean_comparison.py \
        --in_json  figures/clean_comparison.json \
        --out_dir  figures/clean_plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


SCHED_ORDER = [
    "fixed-4", "fixed-8", "fixed-16", "fixed-32",
    "adablock",
    "ours-teacher",
    "ours-oracle",
]

COLOR = {
    "fixed-4":      "#cccccc",
    "fixed-8":      "#999999",
    "fixed-16":     "#666666",
    "fixed-32":     "#333333",
    "adablock":     "#ff7f0e",
    "ours-teacher": "#1f77b4",
    "ours-oracle":  "#d62728",
}


def order_key(label: str) -> int:
    base = label.split(":", 1)[0]
    if base in SCHED_ORDER:
        return SCHED_ORDER.index(base)
    return 99


def fig_per_cell(cell_name: str, cell: Dict, out_path: Path) -> None:
    schedulers = sorted(cell["cells"].keys(), key=order_key)
    accs = [cell["cells"][s]["accuracy"] for s in schedulers]
    speeds = [cell["cells"][s]["tokens_per_second"] for s in schedulers]
    colors = [COLOR.get(s.split(":", 1)[0], "#888888") for s in schedulers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # accuracy
    bars1 = ax1.bar(schedulers, accs, color=colors, edgecolor="black", linewidth=0.6)
    # outline ours-* that beat adablock
    if "adablock" in schedulers:
        ada_acc = accs[schedulers.index("adablock")]
        for b, s, a in zip(bars1, schedulers, accs):
            if s.startswith("ours") and a > ada_acc:
                b.set_edgecolor("#2ca02c")
                b.set_linewidth(2.5)
    for i, v in enumerate(accs):
        ax1.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax1.set_ylim(0, max(accs + [0.01]) * 1.15)
    ax1.set_ylabel("accuracy")
    ax1.set_title(f"Accuracy — {cell_name}")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.tick_params(axis="x", rotation=30)
    for tick in ax1.get_xticklabels():
        tick.set_ha("right")

    # throughput
    bars2 = ax2.bar(schedulers, speeds, color=colors, edgecolor="black", linewidth=0.6)
    if "adablock" in schedulers:
        ada_t = speeds[schedulers.index("adablock")]
        for b, s, t in zip(bars2, schedulers, speeds):
            if s.startswith("ours") and t > ada_t:
                b.set_edgecolor("#2ca02c")
                b.set_linewidth(2.5)
    for i, v in enumerate(speeds):
        ax2.text(i, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax2.set_ylabel("tokens / second")
    ax2.set_title(f"Throughput — {cell_name}")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.tick_params(axis="x", rotation=30)
    for tick in ax2.get_xticklabels():
        tick.set_ha("right")

    fig.suptitle(
        f"{cell_name}  —  prompt-aligned (n = {cell['n_common_prompts']})  "
        f"—  green outline = ours-* beats adablock",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_summary_grid(data: Dict, out_path: Path) -> None:
    """One figure summarizing every (model, benchmark) cell."""
    cells = [(name, c) for name, c in data.items() if len(c["cells"]) >= 4]
    if not cells:
        print("[summary] not enough cells with 4+ schedulers")
        return
    n = len(cells)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.2 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (cell_name, cell) in enumerate(cells):
        ax = axes[idx // cols, idx % cols]
        scheds = sorted(cell["cells"].keys(), key=order_key)
        accs = [cell["cells"][s]["accuracy"] for s in scheds]
        speeds = [cell["cells"][s]["tokens_per_second"] for s in scheds]
        for s, a, t in zip(scheds, accs, speeds):
            base = s.split(":", 1)[0]
            color = COLOR.get(base, "#888")
            marker = "o" if base.startswith("fixed") else ("s" if base.startswith("ours") else "D")
            size = 130 if base.startswith("ours") or base == "adablock" else 80
            ax.scatter(t, a, c=color, marker=marker, s=size,
                       edgecolor="black", linewidth=0.6, label=s, zorder=3)
            ax.annotate(s, (t, a),
                        textcoords="offset points", xytext=(5, 4), fontsize=7)
        # connect fixed-* with a dashed line
        fixed = [(s, accs[scheds.index(s)], speeds[scheds.index(s)])
                 for s in scheds if s.startswith("fixed-")]
        fixed.sort(key=lambda x: int(x[0].split("-")[1]))
        if len(fixed) >= 2:
            xs = [f[2] for f in fixed]
            ys = [f[1] for f in fixed]
            ax.plot(xs, ys, color="#888", linestyle="--", linewidth=1, zorder=1)
        ax.set_xlabel("tokens / second  (faster →)")
        ax.set_ylabel("accuracy  (↑ better)")
        ax.set_title(f"{cell_name}  (n = {cell['n_common_prompts']})")
        ax.grid(True, alpha=0.3)

    # blank any unused panels
    for k in range(len(cells), rows * cols):
        axes[k // cols, k % cols].axis("off")

    fig.suptitle("Clean comparison — accuracy vs throughput  (top-right is best)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_delta_summary(data: Dict, out_path: Path) -> None:
    """Bar chart: Δacc vs adablock per (cell, ours-* scheduler)."""
    cells = [(name, c) for name, c in data.items() if "adablock" in c["cells"]]
    if not cells:
        return

    ours_schedulers = ["ours-teacher", "ours-oracle"]
    fig, ax = plt.subplots(figsize=(11, 5))
    n_cells = len(cells)
    width = 0.35
    x = np.arange(n_cells)
    for i, sched in enumerate(ours_schedulers):
        deltas = []
        for _, c in cells:
            base_acc = c["cells"]["adablock"]["accuracy"]
            if sched in c["cells"]:
                deltas.append(c["cells"][sched]["accuracy"] - base_acc)
            else:
                deltas.append(0.0)
        bars = ax.bar(x + (i - 0.5) * width, deltas, width, color=COLOR[sched],
                      label=sched, edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, deltas):
            if abs(v) > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + (0.001 if v > 0 else -0.001),
                        f"{v:+.3f}", ha="center",
                        va="bottom" if v > 0 else "top", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cells], rotation=20, ha="right")
    ax.set_ylabel("Δ accuracy vs adablock  (positive = ours wins)")
    ax.set_title("Clean comparison: ours-teacher vs ours-oracle vs adablock (prompt-aligned)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in_json", type=Path,
                   default=Path("figures/clean_comparison.json"))
    p.add_argument("--out_dir", type=Path,
                   default=Path("figures/clean_plots"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.in_json.read_text())

    for cell_name, cell in data.items():
        safe = cell_name.replace("/", "_")
        if len(cell["cells"]) >= 4:  # skip degenerate cells
            fig_per_cell(cell_name, cell, args.out_dir / f"{safe}.png")

    fig_summary_grid(data, args.out_dir / "summary_grid.png")
    fig_delta_summary(data, args.out_dir / "delta_vs_adablock.png")


if __name__ == "__main__":
    main()
