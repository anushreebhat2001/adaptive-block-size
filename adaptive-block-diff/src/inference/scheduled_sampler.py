"""Scheduler-driven semi-AR sampler used at eval time.

Wraps the same low-confidence remasking loop as data/runners.py but consults
a Scheduler at every block boundary to choose the next block size.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

from ..data.runners import DiffusionRunner, StepRecord
from .scheduler import SchedulerInput


def _peek_next_window(
    forward_fn,
    seq: torch.Tensor,
    next_window: int,
    mask_id: int,
    pshift: int = 0,
) -> torch.Tensor:
    """Argmax over a peek window appended after the current sequence.

    `pshift` is the model's prediction shift (0 for LLaDA, 1 for Dream).
    Dream's prediction for position i lives at logits[i-1], so the peek read
    starts one position earlier.
    """
    if next_window <= 0:
        return torch.empty(0, dtype=torch.long)
    peek = torch.full((1, next_window), mask_id, dtype=seq.dtype, device=seq.device)
    seq_peek = torch.cat([seq, peek], dim=1)
    with torch.no_grad():
        logits, _ = forward_fn(seq_peek)
    start = seq.shape[1] - pshift
    return (
        logits[0, start : start + next_window]
        .argmax(dim=-1)
        .detach()
        .cpu()
    )


def scheduled_rollout(
    runner: DiffusionRunner,
    prompt_ids: torch.Tensor,
    scheduler,
    max_new_tokens: int,
    n_denoise_steps: int = 32,
    next_window: int = 8,
    temperature: float = 0.0,
    initial_block_size: int = 16,
    min_new_tokens: int = 0,
) -> Tuple[torch.Tensor, List[StepRecord], List[int]]:
    """Decode while letting `scheduler` pick the next block size at each
    boundary.

    Returns:
        generated:    [n] new token ids.
        records:      list[StepRecord] for diagnostics / re-labeling.
        block_sizes:  the actual block sizes chosen, in order.
    """

    forward_fn = runner._forward  # type: ignore[attr-defined]
    device = runner.device
    mask_id = runner.mask_token_id
    eos_id = runner.eos_token_id
    eos_ids = getattr(runner, "eos_token_ids", set()) or {eos_id}
    # Dream uses the AR shift: logits[i] predicts token i+1. We read its
    # predictions from a slice shifted by `pshift` positions. LLaDA leaves
    # this at 0 (predictions live at the masked positions themselves).
    pshift = int(getattr(runner, "prediction_shift", 0))

    seq = prompt_ids.unsqueeze(0).clone()
    records: List[StepRecord] = []
    block_sizes: List[int] = []

    scheduler.reset()
    # Schedulers that don't need a warmup block (e.g. FixedScheduler) can
    # override the rollout's --initial_block_size by exposing one. This makes
    # `fixed-N` actually mean "every block is N" instead of "B=16 first, then
    # N." adablock and the learned schedulers leave this unset and keep the
    # CLI default.
    sched_initial = getattr(scheduler, "initial_block_size", None)
    if sched_initial is not None:
        initial_block_size = int(sched_initial)
    block_size = initial_block_size
    produced = 0
    block_idx = 0

    while produced < max_new_tokens:
        block_len = min(block_size, max_new_tokens - produced)
        if block_len <= 0:
            break
        prefix_len = seq.shape[1]
        mask_block = torch.full(
            (1, block_len), mask_id, dtype=seq.dtype, device=device
        )
        seq = torch.cat([seq, mask_block], dim=1)
        block_slice = slice(prefix_len, prefix_len + block_len)
        # Read slice for prediction logits: identical to block_slice for LLaDA;
        # shifted left by one for Dream so we read its actual prediction for
        # each masked position.
        read_slice = slice(prefix_len - pshift, prefix_len + block_len - pshift)

        force_no_eos = (produced + block_len) <= min_new_tokens

        for _step in range(n_denoise_steps):
            logits, hidden = forward_fn(seq)
            block_logits = logits[0, read_slice].clone()
            if force_no_eos and eos_ids:
                for tid in eos_ids:
                    block_logits[:, tid] = float("-inf")
            still_masked = (seq[0, block_slice] == mask_id)
            if still_masked.sum().item() == 0:
                break
            if temperature > 0:
                probs = F.softmax(block_logits / max(temperature, 1e-6), dim=-1)
                cand = torch.multinomial(probs, num_samples=1).squeeze(-1)
                conf = probs.gather(-1, cand.unsqueeze(-1)).squeeze(-1)
            else:
                conf, cand = F.softmax(block_logits, dim=-1).max(dim=-1)
            n_to_commit = max(
                1,
                int(still_masked.sum().item() // max(1, n_denoise_steps - _step)),
            )
            conf_masked = conf.clone()
            conf_masked[~still_masked] = -1.0
            top_pos = torch.topk(
                conf_masked, k=min(n_to_commit, int(still_masked.sum().item()))
            ).indices
            new_block = seq[0, block_slice].clone()
            new_block[top_pos] = cand[top_pos]
            seq[0, block_slice] = new_block

        # Block-tokens are already committed by the denoise loop and live on
        # `seq`; copying them is free. Logits / hidden / next-window peek
        # require additional forward passes and are only needed by schedulers
        # that actually read SchedulerInput. Skip for FixedScheduler (and any
        # future stateless scheduler) so its throughput measurement reflects
        # only the work the policy needs.
        block_tokens_final = seq[0, block_slice].detach().cpu()
        needs_state = getattr(scheduler, "needs_state", True)
        if needs_state:
            with torch.no_grad():
                logits, hidden = forward_fn(seq)
            # read_slice for logits (Dream shift); hidden stays at block_slice.
            block_logits_final = logits[0, read_slice].detach().to(torch.float32).cpu()
            block_hidden_final = hidden[0, block_slice].detach().to(torch.float32).cpu()
            next_window_ids = _peek_next_window(
                forward_fn, seq, next_window, mask_id, pshift=pshift
            )
        else:
            block_logits_final = torch.empty(0, dtype=torch.float32)
            block_hidden_final = torch.empty(0, dtype=torch.float32)
            next_window_ids = torch.empty(0, dtype=torch.long)

        rec = StepRecord(
            block_index=block_idx,
            position=produced,
            block_logits=block_logits_final,
            block_hidden=block_hidden_final,
            block_token_ids=block_tokens_final,
            next_window_token_ids=next_window_ids,
        )
        records.append(rec)
        block_sizes.append(block_len)

        # Stop only once we're past the min-length window AND the block is
        # dominated by EOS-like tokens. A single inline EOS doesn't end gen.
        past_min_len = (produced + block_len) >= min_new_tokens
        eos_count = sum(int((block_tokens_final == tid).sum().item()) for tid in eos_ids)
        if past_min_len and eos_count > block_len // 2:
            break

        produced += block_len
        block_idx += 1

        if produced < max_new_tokens:
            block_size = scheduler.next_block_size(
                SchedulerInput(
                    block_logits=block_logits_final,
                    block_hidden=block_hidden_final,
                    block_token_ids=block_tokens_final,
                    next_window_token_ids=next_window_ids,
                    position=produced,
                )
            )

    generated = seq[0, prompt_ids.shape[0] :].detach().cpu()
    return generated, records, block_sizes
