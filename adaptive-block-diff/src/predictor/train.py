"""Supervised training for the block-size predictor.

Reads cached (state, label) shards produced by build_teacher_labels.py or
build_oracle_labels.py and trains a BlockSizePredictor.

Designed to run as an HPC sbatch job (see slurm/03_train_predictor.sbatch):
- catches SIGUSR1 (sent by SLURM 2 minutes before timeout) and writes a
  checkpoint before exiting cleanly so requeue can resume.
- supports --resume to pick up from the latest checkpoint.

Example:
    python -m src.predictor.train \
        --shard_glob "/scratch/$USER/labels/oracle/llada/gsm8k/*.pt" \
        --out_ckpt   "/scratch/$USER/ckpts/predictor/llada_oracle.pt" \
        --label_source oracle --model llada
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import PredictorDataset, class_weights
from .features import CANDIDATE_BLOCK_SIZES, n_scalar_features
from .model import BlockSizePredictor, load_predictor, save_predictor


_SHOULD_CHECKPOINT_AND_EXIT = False


def _install_usr1_handler() -> None:
    def handler(signum, frame):
        global _SHOULD_CHECKPOINT_AND_EXIT
        _SHOULD_CHECKPOINT_AND_EXIT = True
        print("[train] received SIGUSR1; will checkpoint after current step", flush=True)

    signal.signal(signal.SIGUSR1, handler)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shard_glob", required=True)
    p.add_argument("--out_ckpt", required=True)
    p.add_argument("--label_source", choices=["teacher", "oracle"], required=True)
    p.add_argument("--model", choices=["llada", "dream"], required=True)
    p.add_argument("--hidden_dim", type=int, required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--mlp_hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--reg_head", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def evaluate(
    model: BlockSizePredictor,
    loader: DataLoader,
    device: str,
    n_classes: int,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    off_by_one_correct = 0
    loss_sum = 0.0
    label_counts = [0] * n_classes
    pred_counts = [0] * n_classes
    with torch.no_grad():
        for scalars, hidden, labels in loader:
            scalars = scalars.to(device, non_blocking=True)
            hidden = hidden.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, _ = model(scalars, hidden)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            preds = logits.argmax(dim=-1)
            total += labels.shape[0]
            correct += (preds == labels).sum().item()
            off_by_one_correct += ((preds - labels).abs() <= 1).sum().item()
            loss_sum += loss.item()
            for c in range(n_classes):
                label_counts[c] += int((labels == c).sum().item())
                pred_counts[c] += int((preds == c).sum().item())
    if total == 0:
        return {
            "val_loss": float("nan"),
            "val_acc": float("nan"),
            "val_off1": float("nan"),
            "val_majority_acc": float("nan"),
            "val_label_dist": label_counts,
            "val_pred_dist": pred_counts,
        }
    majority_count = max(label_counts) if label_counts else 0
    return {
        "val_loss": loss_sum / total,
        "val_acc": correct / total,
        "val_off1": off_by_one_correct / total,
        "val_majority_acc": majority_count / total,
        "val_label_dist": label_counts,
        "val_pred_dist": pred_counts,
    }


def main() -> None:
    args = parse_args()
    _install_usr1_handler()
    torch.manual_seed(args.seed)

    print(f"[train] loading shards from {args.shard_glob}", flush=True)
    train_ds = PredictorDataset(
        args.shard_glob, split="train", val_frac=args.val_frac, max_examples=args.max_examples
    )
    val_ds = PredictorDataset(
        args.shard_glob, split="val", val_frac=args.val_frac, max_examples=args.max_examples
    )
    print(
        f"[train] train={len(train_ds)} val={len(val_ds)} "
        f"class_balance={train_ds.class_balance()}",
        flush=True,
    )

    n_classes = len(CANDIDATE_BLOCK_SIZES)
    cw = class_weights(train_ds, n_classes).to(args.device)

    if args.resume and os.path.exists(args.out_ckpt):
        print(f"[train] resuming from {args.out_ckpt}", flush=True)
        model = load_predictor(args.out_ckpt, map_location=args.device).to(args.device)
    else:
        model = BlockSizePredictor(
            hidden_dim=args.hidden_dim,
            proj_dim=args.proj_dim,
            mlp_hidden=args.mlp_hidden,
            dropout=args.dropout,
            n_classes=n_classes,
            with_regression_head=args.reg_head,
        ).to(args.device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_ds) // args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * steps_per_epoch
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    log_path = args.out_ckpt + ".log.jsonl"
    best_val = float("inf")
    step = 0
    t0 = time.time()

    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for scalars, hidden, labels in train_loader:
            scalars = scalars.to(args.device, non_blocking=True)
            hidden = hidden.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            logits, _ = model(scalars, hidden)
            loss = F.cross_entropy(logits, labels, weight=cw)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            step += 1

            if step % args.log_every == 0:
                with open(log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "step": step,
                                "epoch": epoch,
                                "lr": sched.get_last_lr()[0],
                                "train_loss": float(loss.item()),
                                "elapsed_s": time.time() - t0,
                            }
                        )
                        + "\n"
                    )

            if _SHOULD_CHECKPOINT_AND_EXIT:
                save_predictor(model, args.out_ckpt)
                print(f"[train] saved checkpoint to {args.out_ckpt} after SIGUSR1", flush=True)
                sys.exit(0)

        metrics = evaluate(model, val_loader, args.device, n_classes)
        with open(log_path, "a") as f:
            f.write(
                json.dumps(
                    {"step": step, "epoch": epoch, **metrics, "elapsed_s": time.time() - t0}
                )
                + "\n"
            )
        gain = metrics["val_acc"] - metrics["val_majority_acc"]
        print(
            f"[train] epoch={epoch} step={step} val_loss={metrics['val_loss']:.4f} "
            f"val_acc={metrics['val_acc']:.4f} val_off1={metrics['val_off1']:.4f} "
            f"majority={metrics['val_majority_acc']:.4f} gain_over_majority={gain:+.4f}",
            flush=True,
        )
        print(
            f"[train]   val label_dist={metrics['val_label_dist']} "
            f"pred_dist={metrics['val_pred_dist']}",
            flush=True,
        )
        if metrics["val_loss"] < best_val:
            best_val = metrics["val_loss"]
            save_predictor(model, args.out_ckpt)
            print(f"[train] saved new best to {args.out_ckpt}", flush=True)

    print(f"[train] done. best_val_loss={best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
