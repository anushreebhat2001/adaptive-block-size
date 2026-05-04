"""Teacher-label builder: distill AdaBlock-dLLM's chosen block sizes.

For each calibration prompt, we run the diffusion sampler at a single fixed
default block size (B_default=16) and at every boundary call AdaBlock's
delimiter rule on the just-decoded block + the next-window peek. The block
size that AdaBlock would have chosen there is recorded as the label.

This is the cheap "teacher" labeling: one rollout per prompt rather than K.
The predictor distilled from these labels serves as the lower bound that
oracle-label predictors must beat.

Output shard schema is identical to build_oracle_labels.py so dataset.py can
load either source.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from typing import List

import torch

from ..predictor.features import (
    CANDIDATE_BLOCK_SIZES,
    StateBuilder,
    candidate_index,
    llada_default_delimiters,
    dream_default_delimiters,
)
from .benchmarks import iter_prompts
from .runners import DiffusionRunner, build_runner


_SHOULD_CHECKPOINT_AND_EXIT = False


def _install_usr1_handler() -> None:
    def handler(signum, frame):
        global _SHOULD_CHECKPOINT_AND_EXIT
        _SHOULD_CHECKPOINT_AND_EXIT = True
        print("[teacher] received SIGUSR1; will flush shard and exit", flush=True)

    signal.signal(signal.SIGUSR1, handler)


def _adablock_choice(
    block_token_ids: torch.Tensor,
    next_window_token_ids: torch.Tensor,
    delim_set,
    confidence_mean: float,
    threshold: float = 0.9,
) -> int:
    """AdaBlock-dLLM rule, broadened to a window-style trigger.

    The paper's rule fires when a high-confidence delimiter is detected
    *anywhere* in the just-decoded block (sliding window over confidence),
    not only at the last position. Checking only the last token is a
    false-negative trap when the natural sentence break sits mid-block,
    which routes nearly every boundary into the B=8 fallback.
    """
    delim_in_block = any(int(t.item()) in delim_set for t in block_token_ids)
    delim_in_peek = any(int(t.item()) in delim_set for t in next_window_token_ids)

    if delim_in_block and confidence_mean >= threshold:
        return 32
    if delim_in_peek and confidence_mean >= threshold:
        return 16
    if not delim_in_block and not delim_in_peek:
        return 8
    return 4


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
    p.add_argument(
        "--min_new_tokens",
        type=int,
        default=0,
        help="Mask EOS-like tokens until this many tokens have been generated. "
             "Needed for chat-tuned models that emit <|eot_id|> after one "
             "block; without it we get only ~1 boundary per prompt.",
    )
    p.add_argument(
        "--model_id",
        default=None,
        help="Override HF model id (e.g. GSAI-ML/LLaDA-8B for base instead of Instruct).",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--threshold", type=float, default=0.9, help="AdaBlock confidence threshold")
    p.add_argument("--shard_prefix", default="teacher")
    p.add_argument(
        "--debug_n_boundaries",
        type=int,
        default=0,
        help="print (last_is_delim, delim_in_peek, conf_mean, chosen_B) for the "
             "first N boundaries to help diagnose label collapse",
    )
    return p.parse_args()


def _flush_shard(buf: dict, out_dir: str, shard_idx: int, prefix: str, meta: dict) -> str:
    if buf["scalars"]:
        scalars = torch.stack(buf["scalars"], dim=0)
        hidden_pool = torch.stack(buf["hidden_pool"], dim=0)
        labels = torch.tensor(buf["labels"], dtype=torch.long)
        prompt_ids = torch.tensor(buf["prompt_ids"], dtype=torch.long)
    else:
        return ""
    path = os.path.join(out_dir, f"{prefix}_{shard_idx:06d}.pt")
    torch.save(
        {
            "scalars": scalars,
            "hidden_pool": hidden_pool,
            "labels": labels,
            "prompt_ids": prompt_ids,
            "meta": meta,
        },
        path,
    )
    return path


def main() -> None:
    args = parse_args()
    _install_usr1_handler()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    runner: DiffusionRunner = build_runner(
        args.model, device=args.device, dtype=dtype, model_id=args.model_id
    )
    runner.load()
    print(f"[teacher] loaded {args.model}; hidden_dim={runner.hidden_dim}", flush=True)

    if args.model == "llada":
        delim_ids = set(llada_default_delimiters(runner.tokenizer))
    else:
        delim_ids = set(dream_default_delimiters(runner.tokenizer))
    print(f"[teacher] delimiter token-id set size: {len(delim_ids)}", flush=True)

    out_dir = os.path.join(args.out_dir, args.model, args.benchmark)
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "model": args.model,
        "label_source": "teacher",
        "benchmark": args.benchmark,
        "candidate_block_sizes": CANDIDATE_BLOCK_SIZES,
        "hidden_dim": runner.hidden_dim,
        "default_block_size": args.default_block_size,
        "threshold": args.threshold,
    }

    buf = {"scalars": [], "hidden_pool": [], "labels": [], "prompt_ids": []}
    shard_idx = 0
    n_done = 0
    n_boundaries_seen = 0
    t0 = time.time()

    prompts = iter_prompts(args.benchmark, split=args.split, limit=args.prompt_offset + args.n_prompts)
    for prompt in prompts:
        if prompt.prompt_id < args.prompt_offset:
            continue

        builder = StateBuilder(
            hidden_dim=runner.hidden_dim,
            delimiter_token_ids=delim_ids,
            max_length=args.max_new_tokens,
            default_block_size=args.default_block_size,
        )
        if prompt.messages is not None and runner.has_chat_template():
            prompt_ids = runner.encode_messages(prompt.messages)
        else:
            # Base models (no chat template) get the plaintext few-shot
            # rendering from prompt.text.
            prompt_ids = runner.encode_prompt(prompt.text)
        try:
            _, records = runner.rollout(
                prompt_ids=prompt_ids,
                block_size=args.default_block_size,
                max_new_tokens=args.max_new_tokens,
                n_denoise_steps=args.n_denoise_steps,
                next_window=8,
                min_new_tokens=args.min_new_tokens,
            )
        except Exception as e:
            print(f"[teacher] prompt {prompt.prompt_id} failed: {e}", flush=True)
            continue

        eos_id = getattr(runner, "eos_token_id", -1)
        for rec in records:
            # Skip degenerate boundaries: the whole block is EOS padding,
            # which is not a real semantic juncture and pollutes labels.
            if eos_id is not None and (rec.block_token_ids == eos_id).all():
                continue

            builder.record_block(rec.block_hidden)
            state = builder.build_state(
                block_logits=rec.block_logits,
                next_window_token_ids=rec.next_window_token_ids,
                position=rec.position,
            )
            conf_mean = float(state.scalars[0].item())
            chosen = _adablock_choice(
                block_token_ids=rec.block_token_ids,
                next_window_token_ids=rec.next_window_token_ids,
                delim_set=delim_ids,
                confidence_mean=conf_mean,
                threshold=args.threshold,
            )

            if n_boundaries_seen < args.debug_n_boundaries:
                delim_in_block = any(
                    int(t.item()) in delim_ids for t in rec.block_token_ids
                )
                delim_in_peek = any(
                    int(t.item()) in delim_ids for t in rec.next_window_token_ids
                )
                decoded_block = runner.tokenizer.decode(
                    [int(t.item()) for t in rec.block_token_ids],
                    skip_special_tokens=False,
                )
                decoded_block = decoded_block.replace("\n", "\\n")[:64]
                print(
                    f"[teacher.debug] prompt={prompt.prompt_id} block={rec.block_index} "
                    f"pos={rec.position} delim_in_block={delim_in_block} "
                    f"delim_in_peek={delim_in_peek} conf={conf_mean:.3f} "
                    f"chosen=B{chosen} block={decoded_block!r}",
                    flush=True,
                )
                n_boundaries_seen += 1

            buf["scalars"].append(state.scalars)
            buf["hidden_pool"].append(state.hidden_pool)
            buf["labels"].append(candidate_index(chosen))
            buf["prompt_ids"].append(int(prompt.prompt_id))

        n_done += 1
        if n_done % 25 == 0:
            elapsed = time.time() - t0
            print(
                f"[teacher] {args.model}/{args.benchmark} prompts={n_done} "
                f"buf={len(buf['labels'])} elapsed={elapsed:.0f}s",
                flush=True,
            )

        if len(buf["labels"]) >= args.shard_size or _SHOULD_CHECKPOINT_AND_EXIT:
            path = _flush_shard(buf, out_dir, shard_idx, args.shard_prefix, meta)
            if path:
                print(f"[teacher] wrote shard {path} ({len(buf['labels'])} examples)", flush=True)
                shard_idx += 1
                buf = {"scalars": [], "hidden_pool": [], "labels": [], "prompt_ids": []}
            if _SHOULD_CHECKPOINT_AND_EXIT:
                print("[teacher] exiting after SIGUSR1 flush", flush=True)
                sys.exit(0)

    path = _flush_shard(buf, out_dir, shard_idx, args.shard_prefix, meta)
    if path:
        print(f"[teacher] wrote final shard {path} ({len(buf['labels'])} examples)", flush=True)
    print(f"[teacher] done. total prompts={n_done}", flush=True)


if __name__ == "__main__":
    main()
