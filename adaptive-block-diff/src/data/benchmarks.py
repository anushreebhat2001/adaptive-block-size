"""Calibration prompts for label generation, plus eval-set loaders.

Goal: keep prompt loading in one file so the same indices map to the same
prompts across teacher and oracle label generation, and across eval runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from datasets import load_dataset


@dataclass
class Prompt:
    prompt_id: int
    benchmark: str
    text: str
    reference: Optional[str] = None  # gold answer / canonical solution if any
    # Multi-turn chat history for few-shot. When set, the runner applies the
    # chat template over this list directly so each demonstration is its own
    # user/assistant turn rather than being bundled into one user message
    # (which chat-tuned models tend to see through).
    messages: Optional[List[Dict[str, str]]] = None


_GSM8K_SYSTEM = (
    "You are a careful math tutor. For every problem, show your work "
    "step by step before giving the final numeric answer."
)

_GSM8K_SHOTS = [
    (
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast "
        "and bakes muffins with four. She sells the rest at $2 per egg. "
        "How much does she make daily?",
        "Janet has 16 eggs per day. She eats 3 and uses 4 for muffins, "
        "leaving 16 - 3 - 4 = 9 eggs. She sells those 9 eggs at $2 each, "
        "making 9 * 2 = 18 dollars per day. The answer is 18.",
    ),
    (
        "A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts total does it take?",
        "The robe takes 2 bolts of blue fiber. White fiber is half of that, "
        "so 2 / 2 = 1 bolt of white fiber. The total is 2 + 1 = 3 bolts. "
        "The answer is 3.",
    ),
]


def _gsm8k_messages(question: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": _GSM8K_SYSTEM}]
    for q, a in _GSM8K_SHOTS:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": question})
    return msgs


def _format_gsm8k(question: str) -> str:
    # Plaintext fallback when the runner cannot apply a chat template.
    # The real prompting goes through Prompt.messages.
    rendered = f"{_GSM8K_SYSTEM}\n\n"
    for q, a in _GSM8K_SHOTS:
        rendered += f"Problem: {q}\nSolution: {a}\n\n"
    rendered += f"Problem: {question}\nSolution:"
    return rendered


def _format_math(problem: str) -> str:
    return (
        f"{problem}\n\n"
        f"Show your reasoning, then give the final answer in \\boxed{{}}."
    )


_MBPP_SYSTEM = (
    "You are a Python coding assistant. For each problem, briefly explain "
    "your approach, then provide a complete implementation that passes the "
    "given test."
)

_MBPP_SHOTS = [
    (
        "Write a function to find the shared elements from the given two lists.\n"
        "Test: assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
        "Approach: Convert both inputs to sets and take their intersection, "
        "then return as a tuple.\n\n"
        "```python\n"
        "def similar_elements(a, b):\n"
        "    return tuple(set(a) & set(b))\n"
        "```",
    ),
    (
        "Write a function to identify non-prime numbers.\n"
        "Test: assert is_not_prime(2) == False",
        "Approach: Numbers less than 2 are not prime. Otherwise, check "
        "divisibility from 2 up to sqrt(n).\n\n"
        "```python\n"
        "import math\n"
        "def is_not_prime(n):\n"
        "    if n < 2:\n"
        "        return True\n"
        "    for i in range(2, int(math.sqrt(n)) + 1):\n"
        "        if n % i == 0:\n"
        "            return True\n"
        "    return False\n"
        "```",
    ),
]


def _mbpp_messages(prompt: str, test: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": _MBPP_SYSTEM}]
    for q, a in _MBPP_SHOTS:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": f"{prompt}\nTest: {test}"})
    return msgs


def _format_mbpp(prompt: str, test: str) -> str:
    rendered = f"{_MBPP_SYSTEM}\n\n"
    for q, a in _MBPP_SHOTS:
        rendered += f"{q}\n{a}\n\n"
    rendered += f"{prompt}\nTest: {test}"
    return rendered


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
            yield Prompt(
                prompt_id=i,
                benchmark="gsm8k",
                text=_format_gsm8k(ex["question"]),
                reference=ex.get("answer"),
                messages=_gsm8k_messages(ex["question"]),
            )
    elif bm == "math":
        # The original `hendrycks/competition_math` was pulled from the Hub.
        # Try several known re-hosts in order. All preserve the
        # {problem, solution} schema we need.
        candidates = [
            # (path, config_or_None)
            ("qwedsacf/competition_math", None),
            ("EleutherAI/hendrycks_math", "all"),
            ("nlile/hendrycks-MATH-benchmark", None),
            ("hendrycks/competition_math", None),
        ]
        ds = None
        last_err: Optional[Exception] = None
        for path, config in candidates:
            try:
                ds = (
                    load_dataset(path, config, split=split)
                    if config is not None
                    else load_dataset(path, split=split)
                )
                break
            except Exception as e:
                last_err = e
                continue
        if ds is None:
            raise RuntimeError(
                f"could not load MATH dataset from any known mirror: {last_err}"
            )
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
                messages=_mbpp_messages(ex["text"], test),
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
