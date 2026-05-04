"""Quick sanity check on predictor training results.

Compares the actual final-epoch metrics against what the lambda sweep
predicted at the same lambda, and shows the per-epoch trajectory so you
can verify training was healthy (loss decreasing, no class collapse).

Usage:
    python -m scripts.check_training
"""

from __future__ import annotations

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    user = os.environ.get("USER", "")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt_dir",
        default=f"/scratch/{user}/Efficient-AI/ckpts/predictor",
    )
    p.add_argument(
        "--sweep_json",
        default=f"/scratch/{user}/Efficient-AI/results/lambda_sweep_llada_oracle.json",
    )
    p.add_argument(
        "--lam", type=float, default=0.01, help="lambda to compare against in the sweep"
    )
    return p.parse_args()


def load_jsonl(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def show_trajectory(name: str, log_path: str) -> None:
    rows = [r for r in load_jsonl(log_path) if "val_acc" in r]
    if not rows:
        print(f"  no val rows in {log_path}")
        return
    print(f"\n=== {name} predictor: per-epoch trajectory ===")
    print(
        f"{'epoch':>5} {'step':>5} {'val_loss':>9} {'val_acc':>8} "
        f"{'majority':>9} {'gain':>8} {'off1':>6}  label_dist               pred_dist"
    )
    for r in rows:
        gain = r["val_acc"] - r["val_majority_acc"]
        print(
            f"{r['epoch']:>5} {r['step']:>5} {r['val_loss']:>9.4f} "
            f"{r['val_acc']:>8.4f} {r['val_majority_acc']:>9.4f} "
            f"{gain:>+8.4f} {r.get('val_off1', 0):>6.3f}  "
            f"{str(r['val_label_dist']):<24} {r['val_pred_dist']}"
        )


def compare_to_sweep(actual_path: str, sweep_path: str, lam: float) -> None:
    if not os.path.exists(sweep_path):
        print(f"\n(sweep not found at {sweep_path}; skipping comparison)")
        return
    sweep = json.load(open(sweep_path))
    match = None
    for r in sweep["results"]:
        if abs(r["lambda"] - lam) < 1e-9:
            match = r
            break
    if match is None:
        print(f"\n(no sweep entry at lambda={lam})")
        return

    actual = [r for r in load_jsonl(actual_path) if "val_acc" in r]
    if not actual:
        print(f"\n(no val rows in {actual_path})")
        return
    final = actual[-1]
    a_gain = final["val_acc"] - final["val_majority_acc"]

    print(f"\n=== oracle: actual vs sweep at lambda={lam} ===")
    print(f"  sweep   val_acc={match['val_acc']:.4f}  "
          f"gain={match['val_gain_over_majority']:+.4f}  "
          f"ece={match['val_ece']:.4f}  "
          f"n_val={match['val_n']}")
    print(f"  actual  val_acc={final['val_acc']:.4f}  "
          f"gain={a_gain:+.4f}  "
          f"n_val={sum(final['val_label_dist'])}")
    delta = final["val_acc"] - match["val_acc"]
    print(f"  delta   val_acc={delta:+.4f}  "
          f"(negative is expected if more data was added since sweep)")


def main() -> None:
    args = parse_args()
    oracle_log = os.path.join(args.ckpt_dir, "llada_oracle.pt.log.jsonl")
    teacher_log = os.path.join(args.ckpt_dir, "llada_teacher.pt.log.jsonl")

    if os.path.exists(oracle_log):
        show_trajectory("oracle", oracle_log)
    else:
        print(f"oracle log not found: {oracle_log}")
    if os.path.exists(teacher_log):
        show_trajectory("teacher", teacher_log)
    else:
        print(f"teacher log not found: {teacher_log}")

    if os.path.exists(oracle_log):
        compare_to_sweep(oracle_log, args.sweep_json, args.lam)


if __name__ == "__main__":
    main()
