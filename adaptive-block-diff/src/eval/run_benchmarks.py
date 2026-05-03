"""Benchmark driver. Runs one (model, benchmark, scheduler, lambda) cell.

Writes a JSON result file with per-prompt outcomes and aggregate metrics so
plot_pareto.py can produce the headline figure.

Example:
    python -m src.eval.run_benchmarks \
        --model llada --benchmark gsm8k \
        --scheduler ours-oracle \
        --predictor /scratch/$USER/ckpts/predictor/llada_oracle.pt \
        --out /scratch/$USER/results/llada_gsm8k_ours-oracle.json \
        --n_prompts 200 --max_new_tokens 256
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Optional

import torch
from datasets import load_dataset

from ..data.benchmarks import iter_prompts
from ..data.runners import build_runner
from ..inference.scheduled_sampler import scheduled_rollout
from ..inference.scheduler import make_scheduler
from ..predictor.features import dream_default_delimiters, llada_default_delimiters
from ..predictor.model import load_predictor
from . import scoring


_SHOULD_FLUSH_AND_EXIT = False


def _install_usr1_handler() -> None:
    def handler(signum, frame):
        global _SHOULD_FLUSH_AND_EXIT
        _SHOULD_FLUSH_AND_EXIT = True
        print("[eval] received SIGUSR1; will flush results and exit", flush=True)

    signal.signal(signal.SIGUSR1, handler)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["llada", "dream"], required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--n_prompts", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--n_denoise_steps", type=int, default=32)
    p.add_argument(
        "--scheduler",
        required=True,
        help="fixed-{4,8,16,32}, adablock, ours-teacher, ours-oracle",
    )
    p.add_argument("--predictor", default=None)
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--threshold", type=float, default=0.9, help="adablock threshold")
    p.add_argument("--initial_block_size", type=int, default=16)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--ablation", default=None, help="comma-separated feature names to zero out")
    return p.parse_args()


def _load_test_for(benchmark: str):
    bm = benchmark.lower()
    if bm == "humaneval":
        return load_dataset("openai_humaneval", split="test")
    if bm == "mbpp":
        return load_dataset("mbpp", split="test")
    return None


def main() -> None:
    args = parse_args()
    _install_usr1_handler()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    runner = build_runner(args.model, device=args.device, dtype=dtype)
    runner.load()

    if args.model == "llada":
        delim = llada_default_delimiters(runner.tokenizer)
    else:
        delim = dream_default_delimiters(runner.tokenizer)

    predictor = None
    if args.scheduler.startswith("ours"):
        if args.predictor is None:
            raise SystemExit("--predictor required for ours-* schedulers")
        predictor = load_predictor(args.predictor, map_location=args.device)
        if hasattr(predictor, "to"):
            predictor.to(args.device)

    scheduler = make_scheduler(
        args.scheduler,
        predictor=predictor,
        hidden_dim=runner.hidden_dim,
        delimiter_token_ids=delim,
        max_length=args.max_new_tokens,
        default_block_size=args.initial_block_size,
        threshold=args.threshold,
        device=args.device,
    )

    test_ds = _load_test_for(args.benchmark)

    results = []
    n_correct = 0
    n_total = 0
    total_new_tokens = 0
    total_decode_seconds = 0.0
    block_size_histogram = {}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for i, prompt in enumerate(iter_prompts(args.benchmark, split=args.split, limit=args.n_prompts)):
        prompt_ids = runner.encode_prompt(prompt.text)
        torch.cuda.synchronize() if args.device == "cuda" else None
        t0 = time.time()
        try:
            generated_ids, _, block_sizes = scheduled_rollout(
                runner=runner,
                prompt_ids=prompt_ids,
                scheduler=scheduler,
                max_new_tokens=args.max_new_tokens,
                n_denoise_steps=args.n_denoise_steps,
                next_window=8,
                temperature=0.0,
                initial_block_size=args.initial_block_size,
            )
        except Exception as e:
            print(f"[eval] prompt {prompt.prompt_id} failed: {e}", flush=True)
            continue
        torch.cuda.synchronize() if args.device == "cuda" else None
        dt = time.time() - t0

        for B in block_sizes:
            block_size_histogram[B] = block_size_histogram.get(B, 0) + 1

        gen_text = runner.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if args.benchmark == "gsm8k":
            ok = scoring.gsm8k_correct(gen_text, prompt.reference)
        elif args.benchmark == "math":
            ok = scoring.math_correct(gen_text, prompt.reference)
        elif args.benchmark == "humaneval":
            test = test_ds[i]["test"] if test_ds is not None and i < len(test_ds) else None
            ok = scoring.humaneval_correct(prompt.text, gen_text, prompt.reference, test)
        elif args.benchmark == "mbpp":
            test = test_ds[i]["test_list"][0] if test_ds is not None and i < len(test_ds) else None
            ok = scoring.mbpp_correct(prompt.text, gen_text, test)
        elif args.benchmark == "ifeval":
            ok = scoring.ifeval_correct(gen_text, None)
        else:
            ok = False

        n_total += 1
        n_correct += int(ok)
        total_new_tokens += int(generated_ids.numel())
        total_decode_seconds += dt
        results.append(
            {
                "prompt_id": int(prompt.prompt_id),
                "correct": bool(ok),
                "n_new_tokens": int(generated_ids.numel()),
                "decode_seconds": float(dt),
                "block_sizes": [int(b) for b in block_sizes],
            }
        )

        if (i + 1) % 25 == 0:
            print(
                f"[eval] {args.model}/{args.benchmark}/{args.scheduler} "
                f"{n_total} acc={n_correct / max(n_total,1):.3f} "
                f"toks/s={total_new_tokens / max(total_decode_seconds, 1e-9):.1f}",
                flush=True,
            )
        if _SHOULD_FLUSH_AND_EXIT:
            break

    summary = {
        "model": args.model,
        "benchmark": args.benchmark,
        "scheduler": args.scheduler,
        "lambda": args.lam,
        "threshold": args.threshold,
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": n_correct / max(n_total, 1),
        "tokens_per_second": total_new_tokens / max(total_decode_seconds, 1e-9),
        "total_decode_seconds": total_decode_seconds,
        "block_size_histogram": block_size_histogram,
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] wrote {args.out}", flush=True)
    if _SHOULD_FLUSH_AND_EXIT:
        sys.exit(0)


if __name__ == "__main__":
    main()
