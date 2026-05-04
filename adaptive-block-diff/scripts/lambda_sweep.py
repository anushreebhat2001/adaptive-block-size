"""Offline lambda sweep on cached oracle shards.

Oracle shards cache per-L PPLs at every boundary (see build_oracle_labels.py
buf["per_l_ppl"]). That means we can re-derive labels at any lambda value
without re-running the diffusion model -- pure CPU work.

For each lambda in the sweep range this script:
  1. Reloads oracle shards
  2. Re-derives labels: y*(t) = argmax_L (-PPL_L(t) - lambda/L)
  3. Trains a fresh BlockSizePredictor for `--epochs` epochs on a fixed
     prompt-id-keyed train/val split (matches predictor.train's split)
  4. Reports class_balance, val_acc, majority, gain_over_majority,
     pred_dist, and a calibration metric (expected calibration error)

Outputs a JSON summary so you can pick the lambda with the best
quality/calibration trade-off without re-running rollouts.

Usage:
    python -m scripts.lambda_sweep \\
        --shard_glob "/scratch/$USER/Efficient-AI/labels/oracle/llada/*/*.pt" \\
        --hidden_dim 4096 \\
        --out /scratch/$USER/Efficient-AI/results/lambda_sweep_llada_oracle.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.predictor.features import CANDIDATE_BLOCK_SIZES
from src.predictor.model import BlockSizePredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shard_glob", required=True)
    p.add_argument("--hidden_dim", type=int, required=True)
    p.add_argument(
        "--lambdas",
        default="0.0,0.005,0.01,0.05,0.1,0.5,1.0",
        help="Comma-separated lambda values to sweep",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--n_calibration_bins",
        type=int,
        default=10,
        help="Number of confidence bins for ECE calculation",
    )
    p.add_argument("--out", required=True)
    return p.parse_args()


def _prompt_in_val(prompt_id: int, val_frac: float, salt: str = "v2-2026") -> bool:
    """Match PredictorDataset's deterministic split."""
    h = hashlib.sha256(f"{salt}:{prompt_id}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2**64
    return u < val_frac


def load_shards(shard_glob: str) -> Dict[str, torch.Tensor]:
    """Concatenate all shards into in-memory tensors."""
    paths = sorted(glob.glob(shard_glob))
    if not paths:
        raise FileNotFoundError(f"no shards matched: {shard_glob}")

    scalars_chunks: List[torch.Tensor] = []
    hidden_chunks: List[torch.Tensor] = []
    prompt_id_chunks: List[torch.Tensor] = []
    per_l_ppl_chunks: List[torch.Tensor] = []

    for sp in paths:
        shard = torch.load(sp, map_location="cpu")
        if "per_l_ppl" not in shard:
            raise KeyError(
                f"shard {sp} has no 'per_l_ppl' field -- this is an oracle "
                "lambda sweep, only oracle shards (with cached per-L PPLs) "
                "are supported"
            )
        scalars_chunks.append(shard["scalars"].to(torch.float32))
        hidden_chunks.append(shard["hidden_pool"].to(torch.float32))
        prompt_id_chunks.append(shard["prompt_ids"].to(torch.long))
        per_l_ppl_chunks.append(shard["per_l_ppl"].to(torch.float32))

    return {
        "scalars": torch.cat(scalars_chunks, dim=0),
        "hidden": torch.cat(hidden_chunks, dim=0),
        "prompt_ids": torch.cat(prompt_id_chunks, dim=0),
        "per_l_ppl": torch.cat(per_l_ppl_chunks, dim=0),  # [N, K]
    }


def derive_labels(per_l_ppl: torch.Tensor, lam: float) -> torch.Tensor:
    """y*(t) = argmax_L of (-PPL_L - lambda/L). Vectorized over boundaries."""
    L = torch.tensor(CANDIDATE_BLOCK_SIZES, dtype=torch.float32)
    length_penalty = lam / L                               # [K]
    # PPL is finite where the L-th rollout had a record at this position;
    # inf elsewhere -> utility -inf -> never chosen.
    utility = -per_l_ppl - length_penalty.unsqueeze(0)     # [N, K]
    return utility.argmax(dim=-1).long()


def split_by_prompt(
    prompt_ids: torch.Tensor, val_frac: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    in_val = torch.tensor(
        [_prompt_in_val(int(pid), val_frac) for pid in prompt_ids], dtype=torch.bool
    )
    return ~in_val, in_val


def train_one_lambda(
    data: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> Dict:
    train_mask, val_mask = split_by_prompt(data["prompt_ids"], args.val_frac)
    n_classes = len(CANDIDATE_BLOCK_SIZES)

    # Class weights from train split (inverse frequency, matches train.py).
    train_labels = labels[train_mask]
    counts = torch.zeros(n_classes, dtype=torch.float32)
    for c in range(n_classes):
        counts[c] = max(1.0, float((train_labels == c).sum().item()))
    class_w = (counts.sum() / (n_classes * counts)).to(args.device)

    train_ds = TensorDataset(
        data["scalars"][train_mask],
        data["hidden"][train_mask],
        train_labels,
    )
    val_ds = TensorDataset(
        data["scalars"][val_mask],
        data["hidden"][val_mask],
        labels[val_mask],
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        return {"error": "empty train or val split"}

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    torch.manual_seed(args.seed)
    model = BlockSizePredictor(hidden_dim=args.hidden_dim).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, args.epochs * max(1, len(train_loader)))
    )

    best_val_loss = float("inf")
    best_metrics = {}
    for _ in range(args.epochs):
        model.train()
        for s, h, y in train_loader:
            s, h, y = s.to(args.device), h.to(args.device), y.to(args.device)
            logits, _ = model(s, h)
            loss = F.cross_entropy(logits, y, weight=class_w)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

        m = evaluate(model, val_loader, args, n_classes)
        if m["val_loss"] < best_val_loss:
            best_val_loss = m["val_loss"]
            best_metrics = m

    return best_metrics


def evaluate(
    model: BlockSizePredictor,
    loader: DataLoader,
    args: argparse.Namespace,
    n_classes: int,
) -> Dict:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    label_counts = [0] * n_classes
    pred_counts = [0] * n_classes
    # For ECE: bin top-1 confidences and compare bin accuracy to bin confidence
    n_bins = args.n_calibration_bins
    bin_conf_sum = [0.0] * n_bins
    bin_correct = [0] * n_bins
    bin_count = [0] * n_bins

    with torch.no_grad():
        for s, h, y in loader:
            s, h, y = s.to(args.device), h.to(args.device), y.to(args.device)
            logits, _ = model(s, h)
            loss_sum += float(F.cross_entropy(logits, y, reduction="sum").item())
            probs = F.softmax(logits, dim=-1)
            top_conf, preds = probs.max(dim=-1)
            total += y.shape[0]
            correct += int((preds == y).sum().item())
            for c in range(n_classes):
                label_counts[c] += int((y == c).sum().item())
                pred_counts[c] += int((preds == c).sum().item())
            for conf_v, p_v, y_v in zip(top_conf.tolist(), preds.tolist(), y.tolist()):
                b = min(int(conf_v * n_bins), n_bins - 1)
                bin_conf_sum[b] += conf_v
                bin_correct[b] += int(p_v == y_v)
                bin_count[b] += 1

    if total == 0:
        return {"val_loss": float("nan")}

    ece = 0.0
    bins = []
    for b in range(n_bins):
        if bin_count[b] == 0:
            bins.append({"bin": b, "n": 0, "conf": None, "acc": None})
            continue
        bin_conf = bin_conf_sum[b] / bin_count[b]
        bin_acc = bin_correct[b] / bin_count[b]
        ece += (bin_count[b] / total) * abs(bin_acc - bin_conf)
        bins.append({"bin": b, "n": bin_count[b], "conf": bin_conf, "acc": bin_acc})

    majority_count = max(label_counts)
    return {
        "val_loss": loss_sum / total,
        "val_acc": correct / total,
        "val_majority_acc": majority_count / total,
        "val_gain_over_majority": correct / total - majority_count / total,
        "val_label_dist": label_counts,
        "val_pred_dist": pred_counts,
        "val_ece": ece,
        "val_calibration_bins": bins,
        "val_n": total,
    }


def main() -> None:
    args = parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    print(f"[sweep] lambdas={lambdas}", flush=True)

    data = load_shards(args.shard_glob)
    n_total = data["scalars"].shape[0]
    print(f"[sweep] loaded {n_total} boundaries from oracle shards", flush=True)

    results = []
    for lam in lambdas:
        labels = derive_labels(data["per_l_ppl"], lam)
        n_classes = len(CANDIDATE_BLOCK_SIZES)
        balance = {c: int((labels == c).sum().item()) for c in range(n_classes)}
        print(f"[sweep] lambda={lam} class_balance={balance}", flush=True)

        m = train_one_lambda(data, labels, args)
        m["lambda"] = lam
        m["full_class_balance"] = balance
        results.append(m)

        if "error" not in m:
            print(
                f"[sweep] lambda={lam:.4f} "
                f"val_acc={m['val_acc']:.4f} "
                f"majority={m['val_majority_acc']:.4f} "
                f"gain={m['val_gain_over_majority']:+.4f} "
                f"ece={m['val_ece']:.4f} "
                f"pred_dist={m['val_pred_dist']}",
                flush=True,
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"shard_glob": args.shard_glob, "results": results}, f, indent=2)
    print(f"[sweep] wrote {args.out}", flush=True)

    # Find best lambda by gain_over_majority, breaking ties on lower ECE
    valid = [r for r in results if "error" not in r]
    if valid:
        best = max(valid, key=lambda r: (r["val_gain_over_majority"], -r["val_ece"]))
        print(
            f"[sweep] BEST: lambda={best['lambda']} "
            f"gain={best['val_gain_over_majority']:+.4f} "
            f"ece={best['val_ece']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
