"""Oracle-label builder: K-rollout argmax on the utility u(L) = -PPL - lambda/L.

For each prompt we run K rollouts, one per candidate block size B in {4,8,16,32}.
At each candidate boundary t (defined by the *default* B=16 grid -- positions
{16, 32, 48, ...}) we evaluate, for every candidate L:

    PPL_L(t) = exp(  mean_{i in [t, t+L)} -log p_target(token_i | context)  )
    u_L(t)   = -PPL_L(t)  -  lambda * (1 / L)

where p_target is the diffusion model's own per-position softmax over the
just-finalized block. The label is argmax_L u_L(t).

State features at each boundary are taken from the B=default rollout so the
predictor sees the same input it would see at deployment.

Lambda is logged as metadata; downstream training/eval can choose to refit
labels with a different lambda from the cached per-L PPLs (also stored).
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from ..predictor.features import (
    CANDIDATE_BLOCK_SIZES,
    StateBuilder,
    candidate_index,
    llada_default_delimiters,
    dream_default_delimiters,
)
from .benchmarks import iter_prompts
from .runners import DiffusionRunner, StepRecord, build_runner


_SHOULD_CHECKPOINT_AND_EXIT = False


def _install_usr1_handler() -> None:
    def handler(signum, frame):
        global _SHOULD_CHECKPOINT_AND_EXIT
        _SHOULD_CHECKPOINT_AND_EXIT = True
        print("[oracle] received SIGUSR1; will flush shard and exit", flush=True)

    signal.signal(signal.SIGUSR1, handler)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["llada", "dream"], required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--n_prompts", type=int, default=12500)
    p.add_argument("--prompt_offset", type=int, default=0)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--shard_size", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--default_block_size", type=int, default=16)
    p.add_argument("--n_denoise_steps", type=int, default=32)
    p.add_argument("--lam", type=float, default=0.05, help="length-efficiency penalty lambda")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--shard_prefix", default="oracle")
    return p.parse_args()


def _block_ppl_at(records: List[StepRecord], boundary_pos: int, length: int) -> float:
    """Approximate PPL of the block of size `length` starting at boundary_pos
    by reading per-token confidence (top-1 prob) from the per-block records.

    The records were collected at the *boundary* AFTER each block of size
    L is committed; so for the rollout at candidate L, we pick the block whose
    starting position equals boundary_pos.
    """
    target = None
    for rec in records:
        if rec.position == boundary_pos:
            target = rec
            break
    if target is None:
        return float("inf")
    probs = F.softmax(target.block_logits.to(torch.float32), dim=-1)
    chosen = target.block_token_ids.to(torch.long)
    chosen = chosen.clamp_max(probs.shape[-1] - 1)
    p = probs.gather(-1, chosen.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
    nll = -torch.log(p)
    if nll.numel() == 0:
        return float("inf")
    n = min(length, nll.numel())
    return float(torch.exp(nll[:n].mean()).item())


def _flush_shard(buf: dict, out_dir: str, shard_idx: int, prefix: str, meta: dict) -> str:
    if not buf["scalars"]:
        return ""
    scalars = torch.stack(buf["scalars"], dim=0)
    hidden_pool = torch.stack(buf["hidden_pool"], dim=0)
    labels = torch.tensor(buf["labels"], dtype=torch.long)
    prompt_ids = torch.tensor(buf["prompt_ids"], dtype=torch.long)
    per_l_ppl = torch.tensor(buf["per_l_ppl"], dtype=torch.float32)
    path = os.path.join(out_dir, f"{prefix}_{shard_idx:06d}.pt")
    torch.save(
        {
            "scalars": scalars,
            "hidden_pool": hidden_pool,
            "labels": labels,
            "prompt_ids": prompt_ids,
            "per_l_ppl": per_l_ppl,            # [N, K] for refitting lambda offline
            "meta": meta,
        },
        path,
    )
    return path


def main() -> None:
    args = parse_args()
    _install_usr1_handler()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    runner: DiffusionRunner = build_runner(args.model, device=args.device, dtype=dtype)
    runner.load()
    print(f"[oracle] loaded {args.model}; hidden_dim={runner.hidden_dim}", flush=True)

    if args.model == "llada":
        delim_ids = set(llada_default_delimiters(runner.tokenizer))
    else:
        delim_ids = set(dream_default_delimiters(runner.tokenizer))

    out_dir = os.path.join(args.out_dir, args.model, args.benchmark)
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "model": args.model,
        "label_source": "oracle",
        "benchmark": args.benchmark,
        "candidate_block_sizes": CANDIDATE_BLOCK_SIZES,
        "hidden_dim": runner.hidden_dim,
        "default_block_size": args.default_block_size,
        "lambda": args.lam,
    }

    buf = {
        "scalars": [],
        "hidden_pool": [],
        "labels": [],
        "prompt_ids": [],
        "per_l_ppl": [],
    }
    shard_idx = 0
    n_done = 0
    t0 = time.time()

    prompts = iter_prompts(args.benchmark, split=args.split, limit=args.prompt_offset + args.n_prompts)
    for prompt in prompts:
        if prompt.prompt_id < args.prompt_offset:
            continue

        prompt_ids = runner.encode_prompt(prompt.text)

        # Per-candidate rollouts.
        per_b_records: Dict[int, List[StepRecord]] = {}
        ok = True
        for B in CANDIDATE_BLOCK_SIZES:
            try:
                _, recs = runner.rollout(
                    prompt_ids=prompt_ids,
                    block_size=B,
                    max_new_tokens=args.max_new_tokens,
                    n_denoise_steps=args.n_denoise_steps,
                    next_window=8,
                )
                per_b_records[B] = recs
            except Exception as e:
                print(f"[oracle] prompt {prompt.prompt_id} B={B} failed: {e}", flush=True)
                ok = False
                break
        if not ok:
            continue

        # Default-grid rollout records define the boundaries we will label.
        ref_recs = per_b_records[args.default_block_size]
        builder = StateBuilder(
            hidden_dim=runner.hidden_dim,
            delimiter_token_ids=delim_ids,
            max_length=args.max_new_tokens,
            default_block_size=args.default_block_size,
        )

        for rec in ref_recs:
            builder.record_block(rec.block_hidden)
            state = builder.build_state(
                block_logits=rec.block_logits,
                next_window_token_ids=rec.next_window_token_ids,
                position=rec.position,
            )

            # Evaluate utility u(L) at this boundary for each candidate L.
            ppls = []
            for L in CANDIDATE_BLOCK_SIZES:
                ppls.append(_block_ppl_at(per_b_records[L], rec.position, L))
            utilities = [-p - args.lam * (1.0 / L) for p, L in zip(ppls, CANDIDATE_BLOCK_SIZES)]
            label_idx = int(torch.tensor(utilities).argmax().item())

            buf["scalars"].append(state.scalars)
            buf["hidden_pool"].append(state.hidden_pool)
            buf["labels"].append(label_idx)
            buf["prompt_ids"].append(int(prompt.prompt_id))
            buf["per_l_ppl"].append(ppls)

        n_done += 1
        if n_done % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"[oracle] {args.model}/{args.benchmark} prompts={n_done} "
                f"buf={len(buf['labels'])} elapsed={elapsed:.0f}s",
                flush=True,
            )

        if len(buf["labels"]) >= args.shard_size or _SHOULD_CHECKPOINT_AND_EXIT:
            path = _flush_shard(buf, out_dir, shard_idx, args.shard_prefix, meta)
            if path:
                print(f"[oracle] wrote shard {path} ({len(buf['labels'])} examples)", flush=True)
                shard_idx += 1
                buf = {"scalars": [], "hidden_pool": [], "labels": [], "prompt_ids": [], "per_l_ppl": []}
            if _SHOULD_CHECKPOINT_AND_EXIT:
                print("[oracle] exiting after SIGUSR1 flush", flush=True)
                sys.exit(0)

    path = _flush_shard(buf, out_dir, shard_idx, args.shard_prefix, meta)
    if path:
        print(f"[oracle] wrote final shard {path} ({len(buf['labels'])} examples)", flush=True)
    print(f"[oracle] done. total prompts={n_done}", flush=True)


if __name__ == "__main__":
    main()
