"""Diffusion-model runners for LLaDA-8B and Dream-7B.

Self-contained semi-autoregressive masked-diffusion sampling so that label
generation does not depend on the AdaBlock-dLLM fork being present. The
deployment path under inference/ patches into the fork instead.

Both runners follow the same loop:

  1. Append a block of MASK tokens to the running prefix.
  2. Run the model over the full sequence; collect logits + last hidden states.
  3. Iteratively unmask positions with the highest top-1 confidence
     (low-confidence remasking, as in LLaDA's reference implementation).
  4. Repeat for the configured number of denoising steps.
  5. Commit the block and continue with the next.

This is a faithful but minimal re-implementation. Quality may differ from
the AdaBlock-dLLM fork; for the *headline* benchmark numbers the fork should
be used. Label generation only needs the relative ranking across block sizes
to be preserved, which this sampler does.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


@dataclasses.dataclass
class StepRecord:
    """One block-boundary record collected during a rollout."""
    block_index: int
    position: int                   # tokens committed before this block
    block_logits: torch.Tensor      # [block_len, vocab], CPU float32
    block_hidden: torch.Tensor      # [block_len, hidden_dim], CPU float32
    block_token_ids: torch.Tensor   # [block_len], CPU long
    next_window_token_ids: torch.Tensor  # [W], CPU long (argmax of next-window logits)


class DiffusionRunner:
    """Common interface."""

    name: str = "abstract"
    hidden_dim: int = 0
    vocab_size: int = 0
    mask_token_id: int = 0
    eos_token_id: int = 0

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> None:
        self.device = device
        self.dtype = dtype
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        raise NotImplementedError

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        msg = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            # Tokenizer ships without a chat template; fall back to raw text.
            text = prompt
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        return ids[0]

    def rollout(
        self,
        prompt_ids: torch.Tensor,
        block_size: int,
        max_new_tokens: int,
        n_denoise_steps: int = 32,
        next_window: int = 8,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, List[StepRecord]]:
        """Run a semi-AR rollout. Returns the generated token ids and a list
        of per-block records suitable for feature extraction."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Shared low-confidence-remasking sampler                                     #
# --------------------------------------------------------------------------- #


def _semi_ar_sample(
    *,
    runner: DiffusionRunner,
    prompt_ids: torch.Tensor,
    block_size: int,
    max_new_tokens: int,
    n_denoise_steps: int,
    next_window: int,
    temperature: float,
    forward_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, List[StepRecord]]:
    """Implements the LLaDA-style block-by-block masked diffusion sampler.

    forward_fn(input_ids) -> (logits[1, T, V], hidden[1, T, H])
    """
    device = runner.device
    mask_id = runner.mask_token_id
    eos_id = runner.eos_token_id

    seq = prompt_ids.unsqueeze(0).clone()                # [1, prompt_len]
    records: List[StepRecord] = []
    n_blocks = (max_new_tokens + block_size - 1) // block_size

    for b in range(n_blocks):
        block_len = min(block_size, max_new_tokens - b * block_size)
        if block_len <= 0:
            break
        prefix_len = seq.shape[1]
        mask_block = torch.full(
            (1, block_len), mask_id, dtype=seq.dtype, device=device
        )
        seq = torch.cat([seq, mask_block], dim=1)
        block_slice = slice(prefix_len, prefix_len + block_len)

        # Iterative low-confidence remasking within this block.
        for _step in range(n_denoise_steps):
            logits, hidden = forward_fn(seq)
            block_logits = logits[0, block_slice]            # [block_len, V]
            still_masked = (seq[0, block_slice] == mask_id)
            if still_masked.sum().item() == 0:
                break

            if temperature > 0:
                probs = F.softmax(block_logits / max(temperature, 1e-6), dim=-1)
                cand = torch.multinomial(probs, num_samples=1).squeeze(-1)
                conf = probs.gather(-1, cand.unsqueeze(-1)).squeeze(-1)
            else:
                conf, cand = F.softmax(block_logits, dim=-1).max(dim=-1)

            # Commit the highest-confidence still-masked positions.
            n_to_commit = max(1, int(still_masked.sum().item() // max(1, n_denoise_steps - _step)))
            conf_masked = conf.clone()
            conf_masked[~still_masked] = -1.0
            top_pos = torch.topk(conf_masked, k=min(n_to_commit, int(still_masked.sum().item()))).indices
            new_block = seq[0, block_slice].clone()
            new_block[top_pos] = cand[top_pos]
            seq[0, block_slice] = new_block

        # After block is finalized, capture features for this boundary.
        with torch.no_grad():
            logits, hidden = forward_fn(seq)
        block_logits_final = logits[0, block_slice].detach().to(torch.float32).cpu()
        block_hidden_final = hidden[0, block_slice].detach().to(torch.float32).cpu()
        block_tokens_final = seq[0, block_slice].detach().cpu()

        # Predict argmax for the next-window positions (used for delim feature).
        if prefix_len + block_len < seq.shape[1]:
            nw_logits = logits[0, prefix_len + block_len : prefix_len + block_len + next_window]
            next_window_ids = nw_logits.argmax(dim=-1).detach().cpu()
        else:
            # No tokens past block yet: use a peek by appending W masks (cheap).
            peek = torch.full((1, next_window), mask_id, dtype=seq.dtype, device=device)
            seq_peek = torch.cat([seq, peek], dim=1)
            with torch.no_grad():
                logits_peek, _ = forward_fn(seq_peek)
            next_window_ids = (
                logits_peek[0, seq.shape[1] : seq.shape[1] + next_window]
                .argmax(dim=-1)
                .detach()
                .cpu()
            )

        records.append(
            StepRecord(
                block_index=b,
                position=prefix_len - prompt_ids.shape[0],
                block_logits=block_logits_final,
                block_hidden=block_hidden_final,
                block_token_ids=block_tokens_final,
                next_window_token_ids=next_window_ids,
            )
        )

        # Early stop only if the block is dominated by EOS — a single inline
        # EOS doesn't end generation, but a block of mostly-EOS does.
        if (block_tokens_final == eos_id).sum().item() > block_len // 2:
            break

    generated = seq[0, prompt_ids.shape[0] :].detach().cpu()
    # Trim trailing EOS so callers see real generation length.
    if generated.numel() > 0:
        non_eos_mask = generated != eos_id
        if non_eos_mask.any():
            last_non_eos = int(non_eos_mask.nonzero()[-1].item())
            generated = generated[: last_non_eos + 1]
        else:
            generated = generated[:0]
    return generated, records


# --------------------------------------------------------------------------- #
# LLaDA-8B                                                                    #
# --------------------------------------------------------------------------- #


class LLaDARunner(DiffusionRunner):
    name = "llada"

    def __init__(
        self,
        model_id: str = "GSAI-ML/LLaDA-8B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.model_id = model_id

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype=self.dtype
        ).to(self.device).eval()
        self.hidden_dim = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        # LLaDA reference uses 126336 as mask id. Fall back to tokenizer if added.
        self.mask_token_id = getattr(self.model.config, "mask_token_id", 126336)
        self.eos_token_id = self.tokenizer.eos_token_id or 0

    def _forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            out = self.model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        logits = out.logits if hasattr(out, "logits") else out["logits"]
        hidden = out.hidden_states[-1] if hasattr(out, "hidden_states") else out["hidden_states"][-1]
        return logits, hidden

    def rollout(
        self,
        prompt_ids: torch.Tensor,
        block_size: int,
        max_new_tokens: int,
        n_denoise_steps: int = 32,
        next_window: int = 8,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, List[StepRecord]]:
        return _semi_ar_sample(
            runner=self,
            prompt_ids=prompt_ids,
            block_size=block_size,
            max_new_tokens=max_new_tokens,
            n_denoise_steps=n_denoise_steps,
            next_window=next_window,
            temperature=temperature,
            forward_fn=self._forward,
        )


# --------------------------------------------------------------------------- #
# Dream-7B                                                                    #
# --------------------------------------------------------------------------- #


class DreamRunner(DiffusionRunner):
    name = "dream"

    def __init__(
        self,
        model_id: str = "Dream-org/Dream-v0-Instruct-7B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.model_id = model_id

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype=self.dtype
        ).to(self.device).eval()
        self.hidden_dim = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        # Dream uses Qwen tokenizer; mask is registered as a special token.
        self.mask_token_id = getattr(self.model.config, "mask_token_id", 151666)
        self.eos_token_id = self.tokenizer.eos_token_id or 0

    def _forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            out = self.model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        logits = out.logits
        hidden = out.hidden_states[-1]
        return logits, hidden

    def rollout(
        self,
        prompt_ids: torch.Tensor,
        block_size: int,
        max_new_tokens: int,
        n_denoise_steps: int = 32,
        next_window: int = 8,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, List[StepRecord]]:
        return _semi_ar_sample(
            runner=self,
            prompt_ids=prompt_ids,
            block_size=block_size,
            max_new_tokens=max_new_tokens,
            n_denoise_steps=n_denoise_steps,
            next_window=next_window,
            temperature=temperature,
            forward_fn=self._forward,
        )


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #


def build_runner(model: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> DiffusionRunner:
    if model == "llada":
        return LLaDARunner(device=device, dtype=dtype)
    if model == "dream":
        return DreamRunner(device=device, dtype=dtype)
    raise ValueError(f"unknown model: {model}")
