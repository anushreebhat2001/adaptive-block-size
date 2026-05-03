"""Calibration prompts for label generation, plus eval-set loaders.

Goal: keep prompt loading in one file so the same indices map to the same
prompts across teacher and oracle label generation, and across eval runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

from datasets import load_dataset


@dataclass
class Prompt:
    prompt_id: int
    benchmark: str
    text: str
    reference: Optional[str] = None  # gold answer / canonical solution if any


def _format_gsm8k(question: str) -> str:
    # No "Answer:" suffix: with chat-template models that puts the assistant
    # in a forced-continuation mode and tends to produce 1-2 token replies.
    # Asking for step-by-step reasoning gives multi-paragraph generations.
    return f"{question}\n\nThink step by step, then give the final numeric answer."


def _format_math(problem: str) -> str:
    return (
        f"{problem}\n\n"
        f"Show your reasoning, then give the final answer in \\boxed{{}}."
    )


def _format_mbpp(prompt: str, test: str) -> str:
    # Do NOT pre-open a ```python fence. The chat-templated assistant turn
    # would otherwise just close the fence and emit <|eot_id|>, ending
    # generation after a single short block.
    return (
        f"{prompt}\n\n"
        f"Your function must pass this test:\n{test}\n\n"
        f"First explain your approach in 2-3 sentences, "
        f"then write the complete Python implementation."
    )


def _format_humaneval(prompt: str) -> str:
    return prompt  # already a function signature + docstring


def _format_ifeval(prompt: str) -> str:
    return prompt


def iter_prompts(benchmark: str, split: str = "train", limit: Optional[int] = None) -> Iterator[Prompt]:
    bm = benchmark.lower()
    if bm == "gsm8k":
        ds = load_dataset("gsm8k", "main", split=split)
        for i, ex in enumerate(ds):
            if limit is not None and i >= limit:
                break
            yield Prompt(prompt_id=i, benchmark="gsm8k", text=_format_gsm8k(ex["question"]), reference=ex.get("answer"))
    elif bm == "math":
        ds = load_dataset("hendrycks/competition_math", split=split)
        for i, ex in enumerate(ds):
            if limit is not None and i >= limit:
                break
            yield Prompt(prompt_id=i, benchmark="math", text=_format_math(ex["problem"]), reference=ex.get("solution"))
    elif bm == "mbpp":
        ds = load_dataset("mbpp", split=split)
        for i, ex in enumerate(ds):
            if limit is not None and i >= limit:
                break
            test = ex["test_list"][0] if ex.get("test_list") else ""
            yield Prompt(
                prompt_id=i,
                benchmark="mbpp",
                text=_format_mbpp(ex["text"], test),
                reference=ex.get("code"),
            )
    elif bm == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        for i, ex in enumerate(ds):
            if limit is not None and i >= limit:
                break
            yield Prompt(
                prompt_id=i,
                benchmark="humaneval",
                text=_format_humaneval(ex["prompt"]),
                reference=ex.get("canonical_solution"),
            )
    elif bm == "ifeval":
        ds = load_dataset("HuggingFaceH4/ifeval", split="train")
        for i, ex in enumerate(ds):
            if limit is not None and i >= limit:
                break
            yield Prompt(prompt_id=i, benchmark="ifeval", text=_format_ifeval(ex["prompt"]))
    elif bm in ("owt", "openwebtext"):
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        for i, ex in enumerate(ds):
            if limit is not None and i >= limit:
                break
            text = ex["text"][:512]
            yield Prompt(prompt_id=i, benchmark="owt", text=text)
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")


def collect_prompts(benchmark: str, split: str = "train", limit: Optional[int] = None) -> List[Prompt]:
    return list(iter_prompts(benchmark, split=split, limit=limit))
