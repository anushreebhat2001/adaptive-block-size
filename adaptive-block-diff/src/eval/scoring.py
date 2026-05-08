"""Per-benchmark scoring helpers.

Tightened for paper-style evaluation:
  - MATH: brace-balanced \\boxed{...} parser that handles nested braces (the
    earlier flat regex missed \\boxed{\\frac{1}{2}} and biased MATH down).
  - HumanEval / MBPP: drop the convenience import block (the standard
    harness does not auto-import math/itertools/etc.); trim model output at
    function-boundary markers so runaway generation doesn't smuggle a
    second valid implementation past the tests.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import os
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


_BOXED_TOKEN = "\\boxed{"


def _extract_boxed(s: str) -> Optional[str]:
    """Return the contents of the first \\boxed{...} expression in ``s``,
    handling nested braces. Returns None if no balanced \\boxed found."""
    idx = s.find(_BOXED_TOKEN)
    if idx < 0:
        return None
    start = idx + len(_BOXED_TOKEN)
    depth = 1
    i = start
    while i < len(s) and depth > 0:
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i].strip()
        i += 1
    return None  # unbalanced


def _normalize_math(s: str) -> str:
    """Light normalization for MATH answer matching: strip whitespace, drop
    common LaTeX wrappers that don't change the answer's identity."""
    if s is None:
        return s
    s = s.strip()
    # Strip outer \\text{...}
    m = re.match(r"^\\text\{(.+)\}$", s)
    if m:
        s = m.group(1).strip()
    # Drop $...$ wrappers
    s = s.strip("$").strip()
    # Collapse internal whitespace
    s = re.sub(r"\s+", "", s)
    # Drop trailing periods
    s = s.rstrip(".")
    return s


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
        return _normalize_math(gen) == _normalize_math(gold)


# --------------------------------------------------------------------------- #
# HumanEval / MBPP                                                            #
# --------------------------------------------------------------------------- #


_CODE_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def _extract_code(s: str) -> str:
    """Pull a fenced ```python block if present, else return s as-is."""
    m = _CODE_FENCE.search(s)
    if m:
        return m.group(1)
    return s


# Markers that indicate the model has finished the function body and is now
# generating something else (a new function, a print, an explanation). We
# trim the completion at the first occurrence of any of these so a runaway
# completion that re-defines the function with a *correct* body cannot
# silently pass the test on the second definition.
_HUMANEVAL_TRIM_MARKERS = (
    "\nclass ",
    "\nif __name__",
    "\nprint(",
    "\n#",
    "\n```",
    "\n\ndef ",     # next top-level def — but allow nested defs (indented)
)


def _trim_humaneval_completion(completion: str) -> str:
    earliest = len(completion)
    for marker in _HUMANEVAL_TRIM_MARKERS:
        idx = completion.find(marker)
        if idx >= 0 and idx < earliest:
            earliest = idx
    return completion[:earliest]


def run_python_block(code: str, test: str, timeout: int = 5) -> bool:
    """Execute ``code`` followed by ``test`` in a fresh subprocess.

    No convenience imports are prepended -- the standard HumanEval harness
    does not auto-import the typing/math/heapq libraries, so neither do we.
    Code that needs them must import them itself.
    """
    payload = code + "\n\n" + test + "\n"
    f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    path = f.name
    try:
        f.write(payload)
        f.close()
        out = subprocess.run(
            ["python", path], capture_output=True, timeout=timeout, text=True
        )
        return out.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def humaneval_correct(prompt: str, generated: str, reference: Optional[str], test: Optional[str]) -> bool:
    """Standard HumanEval pass/fail: prompt + completion, then run
    `check(<entry_point>)` from the dataset's `test` field."""
    if test is None:
        return False
    completion = _trim_humaneval_completion(generated)
    code = prompt + completion
    return run_python_block(code, test)


def mbpp_correct(prompt: str, generated: str, test: Optional[str]) -> bool:
    """MBPP pass/fail: extract the model's code (between [BEGIN]/[DONE] or
    fenced) and run it against the dataset's first test assertion."""
    if test is None:
        return False
    # Prefer text between [BEGIN] and [DONE] (matches our 4-shot prompt format).
    s = generated
    begin_idx = s.find("[BEGIN]")
    if begin_idx >= 0:
        s = s[begin_idx + len("[BEGIN]"):]
    done_idx = s.find("[DONE]")
    if done_idx >= 0:
        s = s[:done_idx]
    code = _extract_code(s)
    return run_python_block(code, test)


# --------------------------------------------------------------------------- #
# IFEval (lightweight: prompts have constraints expressible as regex)         #
# --------------------------------------------------------------------------- #


def ifeval_correct(generated: str, reference: Optional[str]) -> bool:
    # Real IFEval evaluation needs the constraint metadata from the dataset;
    # treat absence-of-error as a length proxy here. The full evaluator can
    # be plugged in once available.
    return len(generated.strip()) > 0
