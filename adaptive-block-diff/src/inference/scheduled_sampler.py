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
) -> torch.Tensor:
    """Argmax over a peek window appended after the current sequence."""
    if next_window <= 0:
        return torch.empty(0, dtype=torch.long)
    peek = torch.full((1, next_window), mask_id, dtype=seq.dtype, device=seq.device)
    seq_peek = torch.cat([seq, peek], dim=1)
    with torch.no_grad():
        logits, _ = forward_fn(seq_peek)
    return (
        logits[0, seq.shape[1] : seq.shape[1] + next_window]
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

    seq = prompt_ids.unsqueeze(0).clone()
    records: List[StepRecord] = []
    block_sizes: List[int] = []

    scheduler.reset()
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

        for _step in range(n_denoise_steps):
            logits, hidden = forward_fn(seq)
            block_logits = logits[0, block_slice]
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

        with torch.no_grad():
            logits, hidden = forward_fn(seq)
        block_logits_final = logits[0, block_slice].detach().to(torch.float32).cpu()
        block_hidden_final = hidden[0, block_slice].detach().to(torch.float32).cpu()
        block_tokens_final = seq[0, block_slice].detach().cpu()
        next_window_ids = _peek_next_window(forward_fn, seq, next_window, mask_id)

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

        if (block_tokens_final == eos_id).any():
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
