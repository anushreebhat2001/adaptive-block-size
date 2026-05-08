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
from typing import Callable, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def _collect_eos_token_ids(tokenizer) -> Set[int]:
    """All token ids that should be treated as turn/sequence terminators.

    Llama-3 family emits both ``<|eot_id|>`` (turn end) and ``<|end_of_text|>``;
    Qwen family uses ``<|im_end|>`` and ``<|endoftext|>``. We mask all of them
    when forcing ``min_new_tokens``, otherwise the model just substitutes the
    other terminator and we still get one-block generations.
    """
    ids: Set[int] = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.add(int(tokenizer.eos_token_id))
    candidates = (
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|endoftext|>",
        "<|im_end|>",
    )
    unk = getattr(tokenizer, "unk_token_id", None)
    for tok in candidates:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
        except Exception:
            continue
        if isinstance(tid, int) and tid >= 0 and tid != unk:
            ids.add(tid)
    return ids


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
    # Read offset for predictions. Dream uses the AR shift inherited from
    # Qwen2.5: logits[i] predicts position i+1 (prediction_shift = 1). LLaDA
    # was trained from scratch with logits[i] predicting position i directly
    # (prediction_shift = 0). Without this, Dream's outputs are off-by-one and
    # produce 0% accuracy on every benchmark.
    prediction_shift: int = 0

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> None:
        self.device = device
        self.dtype = dtype
        self.tokenizer = None
        self.model = None
        self.eos_token_ids: Set[int] = set()

    def load(self) -> None:
        raise NotImplementedError

    def encode_prompt(self, prompt: str, raw: bool = False) -> torch.Tensor:
        # When raw=True, send the prompt as plain text (no chat-template wrap).
        # This is the right mode for base models with paper-style few-shot
        # demonstrations: the demos already contain the format, and wrapping in
        # a chat template would put base models out of distribution.
        if not raw and getattr(self.tokenizer, "chat_template", None) is not None:
            msg = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False
            )
        else:
            text = prompt
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        return ids[0]

    def encode_messages(self, messages) -> torch.Tensor:
        """Encode a multi-turn conversation history.

        Each item in ``messages`` is a {"role", "content"} dict. The chat
        template wraps each turn separately, so few-shot demonstrations
        appear as real prior conversation turns rather than being bundled
        into one user message (which chat-tuned models read through).
        """
        try:
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            # Render to a flat string as a last resort.
            text = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        return ids[0]

    def has_chat_template(self) -> bool:
        """True if the tokenizer ships a chat template (Instruct models do;
        Base models don't). Callers use this to decide between the multi-turn
        chat path and the plaintext few-shot path."""
        return getattr(self.tokenizer, "chat_template", None) is not None

    def rollout(
        self,
        prompt_ids: torch.Tensor,
        block_size: int,
        max_new_tokens: int,
        n_denoise_steps: int = 32,
        next_window: int = 8,
        temperature: float = 0.0,
        min_new_tokens: int = 0,
    ) -> Tuple[torch.Tensor, List[StepRecord]]:
        """Run a semi-AR rollout. Returns the generated token ids and a list
        of per-block records suitable for feature extraction.

        ``min_new_tokens``: until this many tokens have been committed past
        the prompt, EOS-like tokens are masked out of the logits so the model
        cannot terminate early. Needed because LLaDA-Instruct otherwise
        emits ``<|eot_id|>`` after one block on most short-answer benchmarks,
        leaving no boundaries for the scheduler to act on.
        """
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
    min_new_tokens: int = 0,
) -> Tuple[torch.Tensor, List[StepRecord]]:
    """Implements the LLaDA-style block-by-block masked diffusion sampler.

    forward_fn(input_ids) -> (logits[1, T, V], hidden[1, T, H])
    """
    device = runner.device
    mask_id = runner.mask_token_id
    eos_id = runner.eos_token_id
    eos_ids: Set[int] = getattr(runner, "eos_token_ids", set()) or {eos_id}
    prompt_len = prompt_ids.shape[0]
    # Dream's logits[i] predict position i+1; LLaDA's logits[i] predict
    # position i. We read the prediction logits from a slice shifted by
    # `pshift` (1 for Dream, 0 for LLaDA). See Dream paper §4.1.
    pshift = int(getattr(runner, "prediction_shift", 0))

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
        # Slice in the logits tensor that holds the predictions for the block.
        # For LLaDA (pshift=0) this equals block_slice. For Dream (pshift=1)
        # the predictions live one position to the left.
        read_slice = slice(prefix_len - pshift, prefix_len + block_len - pshift)

        # If this whole block sits within the min_new_tokens window, EOS-like
        # tokens are forbidden inside it so the model can't terminate early.
        block_end_offset = (prefix_len + block_len) - prompt_len
        force_no_eos = block_end_offset <= min_new_tokens

        # Iterative low-confidence remasking within this block.
        for _step in range(n_denoise_steps):
            logits, hidden = forward_fn(seq)
            block_logits = logits[0, read_slice].clone()    # [block_len, V]
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
        block_logits_final = logits[0, read_slice].detach().to(torch.float32).cpu()
        # Hidden state stays on the block positions: it's used as a learned
        # feature for the predictor regardless of which position the model
        # treats it as predicting.
        block_hidden_final = hidden[0, block_slice].detach().to(torch.float32).cpu()
        block_tokens_final = seq[0, block_slice].detach().cpu()

        # Predict argmax for the next-window positions (used for delim feature).
        if prefix_len + block_len < seq.shape[1]:
            nw_start = prefix_len + block_len - pshift
            nw_logits = logits[0, nw_start : nw_start + next_window]
            next_window_ids = nw_logits.argmax(dim=-1).detach().cpu()
        else:
            # No tokens past block yet: use a peek by appending W masks (cheap).
            peek = torch.full((1, next_window), mask_id, dtype=seq.dtype, device=device)
            seq_peek = torch.cat([seq, peek], dim=1)
            with torch.no_grad():
                logits_peek, _ = forward_fn(seq_peek)
            peek_read_start = seq.shape[1] - pshift
            next_window_ids = (
                logits_peek[0, peek_read_start : peek_read_start + next_window]
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
        self.eos_token_ids = _collect_eos_token_ids(self.tokenizer)

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
        min_new_tokens: int = 0,
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
            min_new_tokens=min_new_tokens,
        )


# --------------------------------------------------------------------------- #
# Dream-7B                                                                    #
# --------------------------------------------------------------------------- #


class DreamRunner(DiffusionRunner):
    name = "dream"
    # Dream is initialized from Qwen2.5 and preserves the AR shift: the
    # hidden state at position i predicts the token at position i+1. See
    # Dream paper §4.1 ("Shift Operation"). The sampler reads logits at
    # position - 1 to recover the prediction for each masked position.
    prediction_shift = 1

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
        # Dream's DreamConfig is not registered with AutoModelForCausalLM
        # (it's a custom diffusion architecture, same situation as LLaDA).
        # AutoModel + trust_remote_code lets the custom modeling_dream.py
        # provide the right class.
        self.model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype=self.dtype
        ).to(self.device).eval()
        self.hidden_dim = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        # Dream uses Qwen tokenizer; mask is registered as a special token.
        self.mask_token_id = getattr(self.model.config, "mask_token_id", 151666)
        self.eos_token_id = self.tokenizer.eos_token_id or 0
        self.eos_token_ids = _collect_eos_token_ids(self.tokenizer)

    def _forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            out = self.model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        logits = out.logits if hasattr(out, "logits") else out["logits"]
        hidden = (
            out.hidden_states[-1]
            if hasattr(out, "hidden_states")
            else out["hidden_states"][-1]
        )
        return logits, hidden

    def rollout(
        self,
        prompt_ids: torch.Tensor,
        block_size: int,
        max_new_tokens: int,
        n_denoise_steps: int = 32,
        next_window: int = 8,
        temperature: float = 0.0,
        min_new_tokens: int = 0,
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
            min_new_tokens=min_new_tokens,
        )


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #


def build_runner(
    model: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    model_id: Optional[str] = None,
) -> DiffusionRunner:
    if model == "llada":
        if model_id is not None:
            return LLaDARunner(model_id=model_id, device=device, dtype=dtype)
        return LLaDARunner(device=device, dtype=dtype)
    if model == "dream":
        if model_id is not None:
            return DreamRunner(model_id=model_id, device=device, dtype=dtype)
        return DreamRunner(device=device, dtype=dtype)
    raise ValueError(f"unknown model: {model}")
