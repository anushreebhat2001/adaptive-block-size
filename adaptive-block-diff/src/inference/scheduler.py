"""Block-size scheduler. Wraps a trained predictor as a drop-in policy.

Three concrete schedulers are provided so eval/run_benchmarks.py can A/B
without changing call sites:

  - FixedScheduler(B)         : returns B at every boundary
  - AdaBlockScheduler(...)    : faithful port of AdaBlock-dLLM's rule
  - LearnedScheduler(predictor, ...)  : our trained MLP

All three share the same interface:
    scheduler.next_block_size(rec_payload) -> int

where rec_payload is a small dict produced by the diffusion sampler at each
boundary -- see SchedulerInput below.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Optional, Sequence

import torch

from ..predictor.features import (
    CANDIDATE_BLOCK_SIZES,
    StateBuilder,
    candidate_from_index,
)
from ..predictor.model import BlockSizePredictor


@dataclasses.dataclass
class SchedulerInput:
    """Everything a scheduler needs at one boundary."""

    block_logits: torch.Tensor          # [block_len, vocab]
    block_hidden: torch.Tensor          # [block_len, hidden_dim]
    block_token_ids: torch.Tensor       # [block_len]
    next_window_token_ids: torch.Tensor # [W]
    position: int


class FixedScheduler:
    def __init__(self, B: int) -> None:
        if B not in CANDIDATE_BLOCK_SIZES:
            raise ValueError(f"unsupported B={B}; must be in {CANDIDATE_BLOCK_SIZES}")
        self.B = B
        self.name = f"fixed-{B}"

    def reset(self) -> None:
        pass

    def next_block_size(self, _: SchedulerInput) -> int:
        return self.B


class AdaBlockScheduler:
    """Faithful port of AdaBlock-dLLM's rule."""

    def __init__(self, delimiter_token_ids: Iterable[int], threshold: float = 0.9) -> None:
        self.delim = set(int(t) for t in delimiter_token_ids)
        self.threshold = threshold
        self.name = "adablock"

    def reset(self) -> None:
        pass

    def next_block_size(self, x: SchedulerInput) -> int:
        delim_in_block = any(int(t.item()) in self.delim for t in x.block_token_ids)
        delim_in_peek = any(int(t.item()) in self.delim for t in x.next_window_token_ids)
        probs = torch.softmax(x.block_logits.to(torch.float32), dim=-1)
        conf = probs.max(dim=-1).values.mean().item()
        if delim_in_block and conf >= self.threshold:
            return 32
        if delim_in_peek and conf >= self.threshold:
            return 16
        if not delim_in_block and not delim_in_peek:
            return 8
        return 4


class LearnedScheduler:
    """Wraps a trained BlockSizePredictor."""

    def __init__(
        self,
        predictor: BlockSizePredictor,
        hidden_dim: int,
        delimiter_token_ids: Iterable[int],
        max_length: int,
        default_block_size: int = 16,
        device: str = "cpu",
        name: str = "ours",
    ) -> None:
        self.predictor = predictor.to(device).eval()
        self.device = device
        self.name = name
        self.builder = StateBuilder(
            hidden_dim=hidden_dim,
            delimiter_token_ids=delimiter_token_ids,
            max_length=max_length,
            default_block_size=default_block_size,
        )

    def reset(self) -> None:
        self.builder = StateBuilder(
            hidden_dim=self.builder.hidden_dim,
            delimiter_token_ids=list(self.builder.delim_set),
            max_length=self.builder.max_length,
            default_block_size=self.builder.default_block_size,
        )

    def next_block_size(self, x: SchedulerInput) -> int:
        self.builder.record_block(x.block_hidden)
        state = self.builder.build_state(
            block_logits=x.block_logits,
            next_window_token_ids=x.next_window_token_ids,
            position=x.position,
        )
        scalars = state.scalars.to(self.device)
        hidden = state.hidden_pool.to(self.device)
        idx = self.predictor.predict_block_size(scalars, hidden)
        # predict_block_size already returns the integer block size, not the index.
        return int(idx)


def make_scheduler(
    kind: str,
    *,
    predictor: Optional[BlockSizePredictor] = None,
    hidden_dim: Optional[int] = None,
    delimiter_token_ids: Optional[Sequence[int]] = None,
    max_length: int = 512,
    default_block_size: int = 16,
    threshold: float = 0.9,
    device: str = "cpu",
):
    """Factory used by run_benchmarks.py."""
    if kind.startswith("fixed-"):
        B = int(kind.split("-", 1)[1])
        return FixedScheduler(B)
    if kind == "adablock":
        if delimiter_token_ids is None:
            raise ValueError("adablock requires delimiter_token_ids")
        return AdaBlockScheduler(delimiter_token_ids, threshold=threshold)
    if kind in ("ours-teacher", "ours-oracle", "ours"):
        if predictor is None or hidden_dim is None or delimiter_token_ids is None:
            raise ValueError("learned scheduler requires predictor, hidden_dim, delimiter_token_ids")
        return LearnedScheduler(
            predictor=predictor,
            hidden_dim=hidden_dim,
            delimiter_token_ids=delimiter_token_ids,
            max_length=max_length,
            default_block_size=default_block_size,
            device=device,
            name=kind,
        )
    raise ValueError(f"unknown scheduler kind: {kind}")
