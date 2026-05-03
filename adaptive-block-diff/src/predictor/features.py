"""State-vector construction for the block-size predictor.

The state at a candidate boundary t is the concatenation of:
  - confidence statistics over the just-decoded block (mean / std / min)
  - top1-top2 margin mean (Prophet convergence signal)
  - per-token entropy (mean / p90)
  - mean-pool of the last hidden states over the previous M=4 blocks
  - boolean: delimiter token in next-W=8 window
  - position fraction, default-block-size one-hot

Two helpers here:
  - extract_step_features(...) : called online during a diffusion rollout to
    produce the scalar features from logits/hidden-states at one boundary.
  - StateBuilder : online accumulator that stores per-block hidden-state pools
    so the M=4 lookback can be computed without re-running the model.

The scalar feature order is fixed (see SCALAR_FEATURE_NAMES) and must stay
stable across label generation, training, and inference.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F


SCALAR_FEATURE_NAMES: List[str] = [
    "conf_mean",
    "conf_std",
    "conf_min",
    "margin_mean",
    "entropy_mean",
    "entropy_p90",
    "delim_in_window",
    "position_frac",
]


CANDIDATE_BLOCK_SIZES: List[int] = [4, 8, 16, 32]


def n_scalar_features() -> int:
    return len(SCALAR_FEATURE_NAMES) + len(CANDIDATE_BLOCK_SIZES)


@dataclasses.dataclass
class StepFeatures:
    """Output of one boundary's feature extraction. All shapes documented."""

    scalars: torch.Tensor          # [n_scalar_features()]
    hidden_pool: torch.Tensor      # [hidden_dim]  (mean over previous M blocks)


class StateBuilder:
    """Online accumulator for per-block features.

    The diffusion sampler should call ``record_block`` after committing each
    block, and ``build_state`` at each candidate boundary.
    """

    def __init__(
        self,
        hidden_dim: int,
        delimiter_token_ids: Sequence[int],
        max_length: int,
        m_lookback: int = 4,
        delim_window: int = 8,
        candidate_block_sizes: Sequence[int] = tuple(CANDIDATE_BLOCK_SIZES),
        default_block_size: int = 16,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.delim_set = set(int(t) for t in delimiter_token_ids)
        self.max_length = max_length
        self.m_lookback = m_lookback
        self.delim_window = delim_window
        self.candidate_block_sizes = list(candidate_block_sizes)
        self.default_block_size = default_block_size
        self._block_hidden_means: List[torch.Tensor] = []

    def record_block(self, block_hidden_states: torch.Tensor) -> None:
        """Push the mean hidden state over a just-committed block.

        Args:
            block_hidden_states: [block_len, hidden_dim] last-layer hidden
                states corresponding to the tokens that were committed in
                this block.
        """
        if block_hidden_states.numel() == 0:
            return
        self._block_hidden_means.append(
            block_hidden_states.detach().mean(dim=0).to(torch.float32).cpu()
        )

    def _hidden_pool(self) -> torch.Tensor:
        if not self._block_hidden_means:
            return torch.zeros(self.hidden_dim, dtype=torch.float32)
        recent = self._block_hidden_means[-self.m_lookback :]
        return torch.stack(recent, dim=0).mean(dim=0)

    def build_state(
        self,
        block_logits: torch.Tensor,
        next_window_token_ids: torch.Tensor,
        position: int,
    ) -> StepFeatures:
        """Build the state vector for one candidate boundary.

        Args:
            block_logits: [block_len, vocab] logits for the most recently
                denoised block, BEFORE argmax/commit. Used for confidence,
                margin, and entropy.
            next_window_token_ids: [<= delim_window] tentative token ids for
                the next W positions (predicted argmax). Used for the
                delimiter-in-window feature.
            position: number of tokens already committed (used for
                position_frac).
        """
        scalars = _scalar_features(
            block_logits=block_logits,
            next_window_token_ids=next_window_token_ids,
            delim_set=self.delim_set,
            max_length=self.max_length,
            position=position,
        )

        block_size_onehot = torch.zeros(len(self.candidate_block_sizes), dtype=torch.float32)
        if self.default_block_size in self.candidate_block_sizes:
            block_size_onehot[self.candidate_block_sizes.index(self.default_block_size)] = 1.0

        scalars = torch.cat([scalars, block_size_onehot], dim=0)
        return StepFeatures(scalars=scalars, hidden_pool=self._hidden_pool())


def _scalar_features(
    block_logits: torch.Tensor,
    next_window_token_ids: torch.Tensor,
    delim_set: set,
    max_length: int,
    position: int,
) -> torch.Tensor:
    """Compute the eight scalar features from the inputs."""
    block_logits = block_logits.detach().to(torch.float32)
    probs = F.softmax(block_logits, dim=-1)

    top2 = torch.topk(probs, k=2, dim=-1).values
    top1 = top2[:, 0]
    top2_runner = top2[:, 1]
    margin = top1 - top2_runner
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

    if next_window_token_ids.numel() == 0:
        delim_in_window = 0.0
    else:
        ids = next_window_token_ids.detach().cpu().tolist()
        delim_in_window = 1.0 if any(int(t) in delim_set for t in ids) else 0.0

    position_frac = float(min(max(position / max(max_length, 1), 0.0), 1.0))

    return torch.tensor(
        [
            top1.mean().item(),
            top1.std(unbiased=False).item() if top1.numel() > 1 else 0.0,
            top1.min().item(),
            margin.mean().item(),
            entropy.mean().item(),
            torch.quantile(entropy, 0.9).item() if entropy.numel() > 0 else 0.0,
            delim_in_window,
            position_frac,
        ],
        dtype=torch.float32,
    )


def candidate_index(block_size: int) -> int:
    """Map a block size to its index in CANDIDATE_BLOCK_SIZES."""
    return CANDIDATE_BLOCK_SIZES.index(block_size)


def candidate_from_index(idx: int) -> int:
    return CANDIDATE_BLOCK_SIZES[idx]


_DELIM_CHARS = (".", "\n", "?", "!", ";", ":")


def _scan_vocab_for_delimiters(tokenizer) -> List[int]:
    """Scan the tokenizer's vocabulary for tokens that are *purely* delimiter
    characters (after stripping leading/trailing whitespace).

    A previous version kept any token whose decoded form merely *contained*
    a delimiter character. That swept in regular words like ``"hello."`` or
    ``"world,"`` and produced ~3000 ids on LLaDA's tokenizer, making the
    "delimiter present" feature fire on almost every block.

    The corrected rule keeps tokens like ``"."``, ``" ."``, ``"\\n"``,
    ``"\\n\\n"``, ``".\\n"``, ``"!"``, ``":"`` -- the ones that actually
    mark sentence/clause boundaries -- and drops the rest.
    """
    delim_chars = set(_DELIM_CHARS)
    ids: List[int] = []
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    for _tok, tok_id in vocab.items():
        try:
            decoded = tokenizer.decode([tok_id], skip_special_tokens=False)
        except Exception:
            continue
        if not decoded:
            continue
        stripped = decoded.strip()
        if not stripped:
            continue
        if all(c in delim_chars for c in stripped):
            ids.append(int(tok_id))
    return sorted(set(ids))


def llada_default_delimiters(tokenizer) -> List[int]:
    """Token ids that count as semantic delimiters for LLaDA's tokenizer."""
    return _scan_vocab_for_delimiters(tokenizer)


def dream_default_delimiters(tokenizer) -> List[int]:
    """Same scan; Dream uses the Qwen tokenizer."""
    return _scan_vocab_for_delimiters(tokenizer)
