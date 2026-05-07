"""Supervised training for the block-size predictor.

Reads cached (state, label) shards produced by build_teacher_labels.py or
build_oracle_labels.py and trains a BlockSizePredictor.

Designed to run as an HPC sbatch job (see slurm/03_train_predictor.sbatch):
- catches SIGUSR1 (sent by SLURM 2 minutes before timeout) and writes a
  checkpoint before exiting cleanly so requeue can resume.
- supports --resume to pick up from the latest checkpoint.

One predictor is trained per (model, label_source). LLaDA and Dream have
different hidden_dim (4096 vs 3584) so they cannot share a single
predictor head -- 03_train_predictor.sbatch handles this by running an
array of jobs, one per (model, source[, lambda]) pair.

Saving criterion: highest val_top1_acc across epochs (NOT lowest val_loss),
because val_loss is unweighted CE and we want the checkpoint that actually
maximizes deployment accuracy under any class-weighting scheme.

Examples (run separately, once per model+source):
    # LLaDA-Base oracle predictor (pooled across 4 benchmarks)
    python -m src.predictor.train \\
        --shard_glob "/scratch/$USER/Efficient-AI/labels/base/oracle/llada/*/*.pt" \\
        --out_ckpt   "/scratch/$USER/Efficient-AI/ckpts/predictor/base/llada_oracle.pt" \\
        --label_source oracle --model llada --hidden_dim 4096
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
from .features import CANDIDATE_BLOCK_SIZES
from .model import BlockSizePredictor, load_predictor, save_predictor


_SHOULD_CHECKPOINT_AND_EXIT = False


def _install_usr1_handler() -> None:
    def handler(_signum, _frame):
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
    p.add_argument(
        "--class_weight_mode",
        choices=["inverse", "sqrt_inverse", "none"],
        default="inverse",
        help="How to weight cross-entropy: 'inverse' (default), 'sqrt_inverse' "
             "(softer correction), or 'none' (no class weights, trust the prior). "
             "All variants use only the data-derived inverse-frequency weights "
             "from class_weights() with no hand-tuned per-class multipliers.",
    )
    p.add_argument(
        "--focal_gamma",
        type=float,
        default=0.0,
        help="Focal loss gamma. 0.0 disables (standard cross-entropy). gamma=2.0 "
             "is the canonical Lin et al. (2017) value for focal loss; it down-"
             "weights well-classified examples and concentrates gradient on hard "
             "examples. Use this instead of hand-tuned class-weight multipliers "
             "when minority classes are systematically underpredicted.",
    )
    p.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing epsilon for cross-entropy. Useful when labels "
             "are noisy (e.g., oracle PPL-based labels). Ignored when "
             "focal_gamma > 0 (focal loss does not currently support smoothing).",
    )
    return p.parse_args()


def focal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss (Lin et al., 2017): (1 - p_t)^gamma * CE(p, y).

    Down-weights examples the model already predicts confidently and up-weights
    hard examples. Equivalent to standard CE when gamma=0.0.
    """
    log_probs = F.log_softmax(logits, dim=-1)             # [B, C]
    log_p_t = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B]
    p_t = log_p_t.exp().clamp_min(1e-12)
    focal = (1.0 - p_t).pow(gamma)
    nll = -log_p_t                                         # standard NLL
    if weight is not None:
        # per-example class weight
        w = weight[labels]
        loss = focal * w * nll
    else:
        loss = focal * nll
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def evaluate(
    model: BlockSizePredictor,
    loader: DataLoader,
    device: str,
    n_classes: int,
    cw: Optional[torch.Tensor] = None,
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
            loss = F.cross_entropy(logits, labels, weight=cw, reduction="sum")
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
    top1 = correct / total
    within1 = off_by_one_correct / total
    return {
        "val_loss": loss_sum / total,
        # primary names
        "val_top1_acc": top1,
        "val_within1_acc": within1,
        # legacy aliases for older log readers
        "val_acc": top1,
        "val_off1": within1,
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

    if args.class_weight_mode == "none":
        cw = None
        print("[train] class_weight_mode=none: using uniform CE", flush=True)
    elif args.class_weight_mode == "sqrt_inverse":
        raw = class_weights(train_ds, n_classes)
        cw = raw.sqrt().to(args.device)
        print(f"[train] class_weight_mode=sqrt_inverse: weights={cw.tolist()}", flush=True)
    else:  # "inverse"
        cw = class_weights(train_ds, n_classes).to(args.device)
        print(f"[train] class_weight_mode=inverse "
              f"(label_source={args.label_source}): weights={cw.tolist()}", flush=True)

    if args.focal_gamma > 0.0:
        print(f"[train] focal_gamma={args.focal_gamma}: using focal loss "
              f"(Lin et al. 2017). label_smoothing ignored.", flush=True)

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
    best_top1 = -1.0
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
            if args.focal_gamma > 0.0:
                loss = focal_cross_entropy(
                    logits, labels, weight=cw, gamma=args.focal_gamma
                )
            else:
                loss = F.cross_entropy(
                    logits,
                    labels,
                    weight=cw,
                    label_smoothing=args.label_smoothing,
                )
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

        metrics = evaluate(model, val_loader, args.device, n_classes, cw=cw)
        with open(log_path, "a") as f:
            f.write(
                json.dumps(
                    {"step": step, "epoch": epoch, **metrics, "elapsed_s": time.time() - t0}
                )
                + "\n"
            )
        gain = metrics["val_top1_acc"] - metrics["val_majority_acc"]
        print(
            f"[train] epoch={epoch} step={step} val_loss={metrics['val_loss']:.4f} "
            f"top1_acc={metrics['val_top1_acc']:.4f} "
            f"within1_acc={metrics['val_within1_acc']:.4f} "
            f"majority={metrics['val_majority_acc']:.4f} gain_over_majority={gain:+.4f}",
            flush=True,
        )
        print(
            f"[train]   val label_dist={metrics['val_label_dist']} "
            f"pred_dist={metrics['val_pred_dist']}",
            flush=True,
        )

        current_top1 = metrics["val_top1_acc"]
        if current_top1 > best_top1:
            best_top1 = current_top1
            save_predictor(model, args.out_ckpt)
            print(f"[train] saved new best to {args.out_ckpt} (top1={current_top1:.4f})", flush=True)

    print(f"[train] done. best_val_top1={best_top1:.4f}", flush=True)


if __name__ == "__main__":
    main()
