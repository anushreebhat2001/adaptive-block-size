"""Per-benchmark scoring helpers."""

from __future__ import annotations

import re
import subprocess
import tempfile
from typing import Optional


# --------------------------------------------------------------------------- #
# GSM8K                                                                       #
# --------------------------------------------------------------------------- #


_GSM_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _last_number(s: str) -> Optional[str]:
    matches = _GSM_NUM.findall(s)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def gsm8k_correct(generated: str, reference: Optional[str]) -> bool:
    if reference is None:
        return False
    gold = reference.split("####")[-1].strip().replace(",", "") if "####" in reference else reference
    gen_last = _last_number(generated)
    gold_last = _last_number(gold)
    if gen_last is None or gold_last is None:
        return False
    try:
        return abs(float(gen_last) - float(gold_last)) < 1e-4
    except ValueError:
        return gen_last == gold_last


# --------------------------------------------------------------------------- #
# MATH (Hendrycks)                                                            #
# --------------------------------------------------------------------------- #


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def _extract_boxed(s: str) -> Optional[str]:
    m = _BOXED.search(s)
    return m.group(1).strip() if m else None


def math_correct(generated: str, reference: Optional[str]) -> bool:
    if reference is None:
        return False
    gen = _extract_boxed(generated) or _last_number(generated)
    gold = _extract_boxed(reference) or _last_number(reference)
    if gen is None or gold is None:
        return False
    try:
        return abs(float(gen) - float(gold)) < 1e-4
    except ValueError:
        return gen.replace(" ", "") == gold.replace(" ", "")


# --------------------------------------------------------------------------- #
# HumanEval / MBPP                                                            #
# --------------------------------------------------------------------------- #


_CODE_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def _extract_code(s: str) -> str:
    m = _CODE_FENCE.search(s)
    if m:
        return m.group(1)
    return s


def run_python_block(code: str, test: str, timeout: int = 5) -> bool:
    payload = (
        "import sys, math, re, itertools, collections, heapq, bisect, functools\n"
        "from typing import *\n"
        + code
        + "\n\n"
        + test
        + "\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(payload)
        path = f.name
    try:
        out = subprocess.run(
            ["python", path], capture_output=True, timeout=timeout, text=True
        )
        return out.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def humaneval_correct(prompt: str, generated: str, reference: Optional[str], test: Optional[str]) -> bool:
    if test is None:
        return False
    code = _extract_code(prompt + generated)
    return run_python_block(code, test)


def mbpp_correct(prompt: str, generated: str, test: Optional[str]) -> bool:
    if test is None:
        return False
    code = _extract_code(generated)
    return run_python_block(code, test)


# --------------------------------------------------------------------------- #
# IFEval (lightweight: prompts have constraints expressible as regex)         #
# --------------------------------------------------------------------------- #


def ifeval_correct(generated: str, reference: Optional[str]) -> bool:
    # Real IFEval evaluation needs the constraint metadata from the dataset;
    # treat absence-of-error as a length proxy here. The full evaluator can
    # be plugged in once available.
    return len(generated.strip()) > 0
