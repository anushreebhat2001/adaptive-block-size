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


_GSM8K_FEW_SHOT = """Solve each problem by showing your work step by step.

Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the rest at $2 per egg. How much does she make daily?
Solution: Janet has 16 eggs per day. She eats 3 and uses 4 for muffins, leaving 16 - 3 - 4 = 9 eggs. She sells these 9 eggs at $2 each, making 9 * 2 = 18 dollars per day. The answer is 18.

Problem: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total does it take?
Solution: The robe takes 2 bolts of blue fiber. White fiber is half of that, so 2 / 2 = 1 bolt of white fiber. The total is 2 + 1 = 3 bolts. The answer is 3.

Problem: {question}
Solution:"""


def _format_gsm8k(question: str) -> str:
    # Few-shot CoT. Without exemplars, LLaDA-Instruct outputs a one-line
    # bottom-line answer and ends the turn -- giving us only 1 boundary
    # per prompt and rendering the scheduler invisible. With 2 worked
    # examples the model continues the pattern and produces multi-step
    # reasoning (~100-200 tokens), exercising the scheduler at 6-12
    # boundaries per prompt.
    return _GSM8K_FEW_SHOT.format(question=question)


def _format_math(problem: str) -> str:
    return (
        f"{problem}\n\n"
        f"Show your reasoning, then give the final answer in \\boxed{{}}."
    )


_MBPP_FEW_SHOT = '''You will be given a Python programming problem and a test it must pass. Briefly explain your approach, then write the implementation.

Problem: Write a function to find the shared elements from the given two lists.
Test: assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
Approach: Convert both inputs to sets and intersect, then return the result as a tuple.
Implementation:
def similar_elements(a, b):
    return tuple(set(a) & set(b))

Problem: Write a function to identify non-prime numbers.
Test: assert is_not_prime(2) == False
Approach: Numbers less than 2 are not prime. Otherwise, check divisibility from 2 to sqrt(n).
Implementation:
import math
def is_not_prime(n):
    if n < 2:
        return True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return True
    return False

Problem: {prompt}
Test: {test}
Approach:'''


def _format_mbpp(prompt: str, test: str) -> str:
    # Few-shot exemplars so LLaDA continues the "Approach: ... Implementation: ..."
    # pattern instead of emitting a 1-line stub and ending the turn.
    return _MBPP_FEW_SHOT.format(prompt=prompt, test=test)


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
