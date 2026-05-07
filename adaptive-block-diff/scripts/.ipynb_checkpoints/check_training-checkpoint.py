"""Inspect predictor training trajectories across all (model, source[, tag]) configs.

Auto-discovers every `*.pt.log.jsonl` file under ckpt_dir, parses the
per-epoch validation rows, and prints:
  1. A compact trajectory per config (best-by-loss + best-by-top1 rows).
  2. A cross-config summary table at the saved checkpoint epoch.
  3. A findings section: oracle vs teacher per model, llada vs dream per source.

Filename convention supported:
    <model>_<source>.pt.log.jsonl              -> tag = "" (the base run)
    <model>_<source>_<tag>.pt.log.jsonl        -> tag = "<tag>"   (e.g. "lam5", "lam10")

Each (model, source, tag) is treated as its own config in the display.
The findings section compares oracle vs teacher within each (model, tag) pair.

Usage:
    python -m scripts.check_training
    python -m scripts.check_training --ckpt_dir /scratch/$USER/Efficient-AI/ckpts/predictor/base
    python -m scripts.check_training --full_trajectory
    python -m scripts.check_training --tag_filter lam5      # only show the lam5 runs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    user = os.environ.get("USER", "")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt_dir",
        default=f"/scratch/{user}/Efficient-AI/ckpts/predictor/base",
        help="Directory containing predictor ckpts and their *.log.jsonl files.",
    )
    p.add_argument(
        "--full_trajectory",
        action="store_true",
        help="Print every epoch (default: only best-loss + best-top1 rows).",
    )
    p.add_argument(
        "--tag_filter",
        default=None,
        help='Only show configs whose tag matches exactly. Use "" to match the '
             'base (untagged) runs only. Default: show all.',
    )
    return p.parse_args()


def load_jsonl(path: str) -> List[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def discover_configs(ckpt_dir: str) -> List[Tuple[str, str, str, str]]:
    """Return list of (model, source, tag, log_path) discovered under ckpt_dir.

    Parses filenames `<model>_<source>[_<tag>].pt.log.jsonl` by locating the
    'teacher' or 'oracle' token anywhere in the underscore-separated name.
    Everything before it is the model id (joined back with underscores if
    multi-word), everything after is the tag. tag is "" for base runs.
    """
    found: List[Tuple[str, str, str, str]] = []
    for path in sorted(glob.glob(os.path.join(ckpt_dir, "*.pt.log.jsonl"))):
        base = os.path.basename(path).replace(".pt.log.jsonl", "")
        parts = base.split("_")
        if len(parts) < 2:
            continue
        src_idx = None
        for i, tok in enumerate(parts):
            if tok in ("teacher", "oracle"):
                src_idx = i
                break
        if src_idx is None:
            # filename doesn't contain teacher/oracle - skip
            continue
        model = "_".join(parts[:src_idx])
        if not model:
            continue
        source = parts[src_idx]
        tag = "_".join(parts[src_idx + 1:])  # "" if nothing after source
        found.append((model, source, tag, path))
    return found


def val_rows(rows: List[dict]) -> List[dict]:
    return [r for r in rows if "val_acc" in r or "val_top1_acc" in r]


def get_top1(r: dict) -> float:
    return r.get("val_top1_acc", r.get("val_acc", float("nan")))


def get_within1(r: dict) -> float:
    return r.get("val_within1_acc", r.get("val_off1", float("nan")))


def best_by_loss(rows: List[dict]) -> Optional[dict]:
    if not rows:
        return None
    return min(rows, key=lambda r: r["val_loss"])


def best_by_top1(rows: List[dict]) -> Optional[dict]:
    if not rows:
        return None
    return max(rows, key=get_top1)


def fmt_row(r: dict) -> str:
    top1 = get_top1(r)
    within1 = get_within1(r)
    gain = top1 - r["val_majority_acc"]
    return (
        f"epoch={r['epoch']:>2}  val_loss={r['val_loss']:.4f}  "
        f"top1_acc={top1:.4f}  within1_acc={within1:.4f}  "
        f"majority={r['val_majority_acc']:.4f}  gain={gain:+.4f}"
    )


def _config_label(model: str, source: str, tag: str) -> str:
    return f"{model} / {source}" + (f" / {tag}" if tag else "")


def print_compact_trajectory(name: str, rows: List[dict]) -> None:
    print(f"\n=== {name} ===")
    if not rows:
        print("  no val rows")
        return
    bl = best_by_loss(rows)
    bt = best_by_top1(rows)
    print(f"  total epochs run : {len(rows)}")
    print(f"  saved ckpt (best val_loss) : {fmt_row(bl)}")
    if bt and (bt["epoch"] != bl["epoch"]):
        print(f"  best top1_acc epoch        : {fmt_row(bt)}")
    print(f"  final epoch                : {fmt_row(rows[-1])}")
    print(f"  val label_dist : {rows[-1]['val_label_dist']}")
    print(f"  saved-ckpt pred_dist : {bl['val_pred_dist']}")


def print_full_trajectory(name: str, rows: List[dict]) -> None:
    print(f"\n=== {name} (full per-epoch) ===")
    if not rows:
        print("  no val rows")
        return
    print(
        f"  {'epoch':>5} {'val_loss':>9} {'top1_acc':>9} {'within1':>9} "
        f"{'majority':>9} {'gain':>8}  label_dist               pred_dist"
    )
    for r in rows:
        top1 = get_top1(r)
        within1 = get_within1(r)
        gain = top1 - r["val_majority_acc"]
        print(
            f"  {r['epoch']:>5} {r['val_loss']:>9.4f} "
            f"{top1:>9.4f} {within1:>9.4f} "
            f"{r['val_majority_acc']:>9.4f} {gain:>+8.4f}  "
            f"{str(r['val_label_dist']):<24} {r['val_pred_dist']}"
        )


CANDIDATE_BLOCK_SIZES = [4, 8, 16, 32]


def _mean_inv_L(dist: List[int]) -> float:
    """Given a count-per-class distribution, return mean 1/L.

    Lower = fewer forward passes per token = faster deployment.
    """
    total = sum(dist)
    if total == 0:
        return float("nan")
    s = 0.0
    for cnt, L in zip(dist, CANDIDATE_BLOCK_SIZES):
        s += cnt * (1.0 / L)
    return s / total


def summary_for(rows: List[dict]) -> Optional[Dict]:
    bl = best_by_loss(rows)
    if bl is None:
        return None
    label_dist = bl["val_label_dist"]
    pred_dist = bl["val_pred_dist"]
    return {
        "epoch": bl["epoch"],
        "val_loss": bl["val_loss"],
        "top1": get_top1(bl),
        "within1": get_within1(bl),
        "majority": bl["val_majority_acc"],
        "gain": get_top1(bl) - bl["val_majority_acc"],
        "label_dist": label_dist,
        "pred_dist": pred_dist,
        "n_val": sum(label_dist),
        # mean 1/L = mean compute proxy. Lower = faster deployment.
        "compute_pred":   _mean_inv_L(pred_dist),
        "compute_labels": _mean_inv_L(label_dist),
    }


def cross_config_table(summaries: Dict[Tuple[str, str, str], Dict]) -> None:
    print("\n" + "=" * 120)
    print("CROSS-CONFIG SUMMARY (saved-checkpoint epoch = best val_loss)")
    print("=" * 120)
    print(
        f"  {'model':<8} {'source':<8} {'tag':<10} {'epoch':>5} {'n_val':>7} "
        f"{'val_loss':>9} {'top1':>7} {'within1':>8} "
        f"{'major':>7} {'gain':>7} {'cmp_pred':>9} {'cmp_lbl':>9}"
    )
    print("  " + "-" * 130)
    for (model, source, tag), s in sorted(summaries.items()):
        print(
            f"  {model:<8} {source:<8} {(tag or '-'):<10} {s['epoch']:>5} {s['n_val']:>7} "
            f"{s['val_loss']:>9.4f} {s['top1']:>7.4f} {s['within1']:>8.4f} "
            f"{s['majority']:>7.4f} {s['gain']:>+7.4f} "
            f"{s['compute_pred']:>9.5f} {s['compute_labels']:>9.5f}"
        )


def findings(summaries: Dict[Tuple[str, str, str], Dict]) -> None:
    print("\n" + "=" * 120)
    print("FINDINGS")
    print("=" * 120)

    models = sorted({m for (m, _, _) in summaries})
    sources = sorted({s for (_, s, _) in summaries})
    tags = sorted({t for (_, _, t) in summaries})

    # 1) Oracle vs Teacher per (model, tag)
    print("\n[1] Oracle vs Teacher per (model, tag)")
    for model in models:
        for tag in tags:
            t = summaries.get((model, "teacher", tag))
            o = summaries.get((model, "oracle", tag))
            if not (t and o):
                continue
            label = f"{model} / tag={tag or '-'}"
            diff_top1 = o["top1"] - t["top1"]
            diff_gain = o["gain"] - t["gain"]
            diff_within1 = o["within1"] - t["within1"]
            if o["gain"] > t["gain"] and o["within1"] >= t["within1"] - 0.02:
                verdict = "ORACLE WINS"
            elif t["gain"] > o["gain"] + 0.05:
                verdict = "TEACHER WINS — unexpected; check label distributions"
            else:
                verdict = "TIE — within noise"
            print(f"  {label:<22}: oracle gain={o['gain']:+.3f}  vs  teacher gain={t['gain']:+.3f}  "
                  f"(Δgain={diff_gain:+.3f}, Δtop1={diff_top1:+.3f}, Δwithin1={diff_within1:+.3f})")
            print(f"            -> {verdict}")
            if t["majority"] - o["majority"] > 0.20:
                print(f"            note: teacher's majority ({t['majority']:.2f}) is much higher than "
                      f"oracle's ({o['majority']:.2f}); teacher's high top1 mostly reflects label "
                      f"imbalance, not learning")

    # 2) LLaDA vs Dream per (source, tag)
    print("\n[2] LLaDA vs Dream per (source, tag)")
    for source in sources:
        for tag in tags:
            l = summaries.get(("llada", source, tag))
            d = summaries.get(("dream", source, tag))
            if not (l and d):
                continue
            label = f"{source} / tag={tag or '-'}"
            diff_gain = d["gain"] - l["gain"]
            diff_top1 = d["top1"] - l["top1"]
            if abs(diff_gain) < 0.02:
                verdict = "GENERALIZES"
            elif l["gain"] > d["gain"] + 0.05:
                verdict = f"LLaDA STRONGER by {l['gain'] - d['gain']:.3f}"
            elif d["gain"] > l["gain"] + 0.05:
                verdict = f"DREAM STRONGER by {d['gain'] - l['gain']:.3f}"
            else:
                verdict = f"SIMILAR (Δgain={diff_gain:+.3f})"
            print(f"  {label:<22}: llada gain={l['gain']:+.3f}  vs  dream gain={d['gain']:+.3f}  "
                  f"(Δgain={diff_gain:+.3f}, Δtop1={diff_top1:+.3f})")
            print(f"            -> {verdict}")

    # 3) Per-tag oracle Pareto over models (for picking the best deployable)
    print("\n[3] Best deployable predictor per tag (oracle source, gain over majority)")
    for tag in tags:
        oracles = {(m, t): summaries[(m, "oracle", t)]
                   for (m, src, t) in summaries
                   if src == "oracle" and t == tag}
        if not oracles:
            continue
        best_key = max(oracles, key=lambda k: oracles[k]["gain"])
        best = oracles[best_key]
        suffix = f"_{tag}" if tag else ""
        print(f"  tag={tag or '-':<10} winner = {best_key[0]} oracle  "
              f"(gain={best['gain']:+.3f}, top1={best['top1']:.3f}, within1={best['within1']:.3f})  "
              f"-> {best_key[0]}_oracle{suffix}.pt")

    # 3b) Deployment-best comparison: best oracle per model vs best teacher per
    #     model, ignoring tag. Useful when oracle is at lam_5/lam_10 but teacher
    #     is at base, so the per-tag pairings in [1] don't fire.
    print("\n[3b] Deployment-best per model (best oracle vs best teacher, any tag)")
    by_model_source: Dict[Tuple[str, str], List[Tuple[str, Dict]]] = {}
    for (model, source, tag), s in summaries.items():
        by_model_source.setdefault((model, source), []).append((tag, s))
    best_oracle: Dict[str, Tuple[str, Dict]] = {}
    best_teacher: Dict[str, Tuple[str, Dict]] = {}
    for (model, source), entries in by_model_source.items():
        # pick the entry with the highest top1 (deployment metric)
        best = max(entries, key=lambda kv: kv[1]["top1"])
        if source == "oracle":
            best_oracle[model] = best
        elif source == "teacher":
            best_teacher[model] = best
    for model in sorted(set(list(best_oracle.keys()) + list(best_teacher.keys()))):
        o_tag, o = best_oracle.get(model, (None, None))
        t_tag, t = best_teacher.get(model, (None, None))
        if o is None or t is None:
            print(f"  {model:<8}: incomplete (need both an oracle and a teacher run)")
            continue
        diff_top1 = o["top1"] - t["top1"]
        diff_within1 = o["within1"] - t["within1"]
        if o["top1"] >= t["top1"] and o["within1"] >= t["within1"] - 0.02:
            verdict = "ORACLE >= TEACHER on top1"
        elif t["top1"] > o["top1"] + 0.05:
            verdict = "TEACHER WINS on top1 (note teacher's higher majority)"
        else:
            verdict = "TIE on top1 within noise"
        print(f"  {model:<8}: best oracle (tag={o_tag or '-'})  top1={o['top1']:.3f}  within1={o['within1']:.3f}  "
              f"vs  best teacher (tag={t_tag or '-'})  top1={t['top1']:.3f}  within1={t['within1']:.3f}")
        print(f"            -> {verdict} (Δtop1={diff_top1:+.3f}, Δwithin1={diff_within1:+.3f})")

    # Cross-model deployment-best comparison
    if len(best_oracle) >= 2:
        print("\n[3c] Deployment-best LLaDA oracle vs Dream oracle (any tag)")
        if "llada" in best_oracle and "dream" in best_oracle:
            l_tag, l = best_oracle["llada"]
            d_tag, d = best_oracle["dream"]
            print(f"  LLaDA  (tag={l_tag or '-'}):  "
                  f"top1={l['top1']:.3f}  within1={l['within1']:.3f}  "
                  f"compute_pred={l['compute_pred']:.5f}  (vs labels {l['compute_labels']:.5f})")
            print(f"  Dream  (tag={d_tag or '-'}):  "
                  f"top1={d['top1']:.3f}  within1={d['within1']:.3f}  "
                  f"compute_pred={d['compute_pred']:.5f}  (vs labels {d['compute_labels']:.5f})")

    # 4) Sanity flags
    print("\n[4] Sanity flags")
    flagged_any = False
    for (model, source, tag), s in sorted(summaries.items()):
        flags = []
        if s["gain"] < 0.01:
            flags.append("gain ≈ 0 (predictor not learning beyond majority)")
        nz = sum(1 for c in s["pred_dist"] if c >= 0.05 * sum(s["pred_dist"]))
        if nz < 3:
            flags.append(f"pred_dist uses only {nz} classes meaningfully (>5% mass)")
        if s["within1"] < 0.7:
            flags.append("within1_acc < 0.7 (predictor's choices aren't even close)")
        if flags:
            flagged_any = True
            label = _config_label(model, source, tag)
            print(f"  {label}: " + "; ".join(flags))
    if not flagged_any:
        print("  no flags — all predictors learning real structure")


def main() -> None:
    args = parse_args()
    print(f"ckpt_dir = {args.ckpt_dir}")
    configs = discover_configs(args.ckpt_dir)
    if args.tag_filter is not None:
        configs = [c for c in configs if c[2] == args.tag_filter]
    if not configs:
        print(f"\nno *.log.jsonl files found under {args.ckpt_dir}"
              f"{f' (after tag_filter={args.tag_filter!r})' if args.tag_filter is not None else ''}")
        return
    print(f"\ndiscovered {len(configs)} config(s):")
    for m, s, t, p in configs:
        try:
            n_rows = sum(1 for _ in open(p))
        except OSError:
            n_rows = -1
        try:
            mtime = os.path.getmtime(p)
            import datetime as _dt
            mtime_str = _dt.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        except OSError:
            mtime_str = "?"
        print(f"  ({m:<6} {s:<8} tag={t or '-':<8})  rows={n_rows:>4}  mtime={mtime_str}  <- {p}")

    rows_by_config: Dict[Tuple[str, str, str], List[dict]] = {}
    summaries: Dict[Tuple[str, str, str], Dict] = {}
    for model, source, tag, path in configs:
        rows = val_rows(load_jsonl(path))
        rows_by_config[(model, source, tag)] = rows
        s = summary_for(rows)
        if s is not None:
            summaries[(model, source, tag)] = s
        name = _config_label(model, source, tag)
        if args.full_trajectory:
            print_full_trajectory(name, rows)
        else:
            print_compact_trajectory(name, rows)

    if summaries:
        cross_config_table(summaries)
        findings(summaries)


if __name__ == "__main__":
    main()
