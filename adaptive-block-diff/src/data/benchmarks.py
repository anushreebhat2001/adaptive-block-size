"""Calibration prompts and eval-set loaders.

Reformulated to follow the Dream-paper / lm-evaluation-harness convention for
**base-model** evaluation:

  - Few-shot counts match Dream paper Table 1 footnotes:
      GSM8K  : 8-shot  (Wei et al. CoT prompts)
      MATH   : 4-shot  (Hendrycks et al. style, \\boxed answers)
      MBPP   : 4-shot  (BEGIN/DONE format)
      HumanEval: 0-shot (raw function signature + docstring)

  - Few-shot demos are concatenated as plain text. The runner's encode_prompt
    is invoked with raw=True so no chat template is applied. This is the right
    setting for base models (LLaDA-Base, Dream-Base): wrapping in a chat
    template puts them out of distribution.

  - Each Prompt now carries a list of `stop_sequences` strings. Generation is
    truncated at the first one that appears in the decoded output. Without
    stop sequences the model continues into a hallucinated next-problem and
    answer extraction picks up the wrong number.

  - The Prompt.messages field (chat path) is left unset in the new format.
    `runner.encode_messages` is no longer called from run_benchmarks.py for
    base-model eval; everything goes through `encode_prompt(..., raw=True)`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from datasets import load_dataset


@dataclass
class Prompt:
    prompt_id: int
    benchmark: str
    text: str
    reference: Optional[str] = None
    # Multi-turn chat history for instruct-mode eval. Unused in base-model
    # raw-text mode but kept for future use.
    messages: Optional[List[Dict[str, str]]] = None
    # Strings at which generation should be truncated. Allows base-model
    # few-shot prompting to avoid bleed into the next problem.
    stop_sequences: List[str] = field(default_factory=list)


# --------------------------------------------------------------------- GSM8K --
# Wei et al. 2022 8-shot CoT demos, the standard reference.

_GSM8K_SHOTS: List[Tuple[str, str]] = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in "
        "the grove today. After they are done, there will be 21 trees. How "
        "many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some "
        "more were planted. So there must have been 21 - 15 = 6. The answer "
        "is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how "
        "many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The "
        "answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how "
        "many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total "
        "they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The "
        "answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has "
        "12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. How many computers "
        "are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = "
        "29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. How many golf balls did he have at the "
        "end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he "
        "had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf "
        "balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money "
        "does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = "
        "15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The "
        "answer is 8.",
    ),
]


def _format_gsm8k(question: str) -> str:
    parts: List[str] = []
    for q, a in _GSM8K_SHOTS:
        parts.append(f"Question: {q}\nAnswer: {a}")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


_GSM8K_STOPS = ["\nQuestion:", "\n\nQuestion:"]


# ---------------------------------------------------------------------- MATH --
# Hendrycks-style 4-shot demos with \boxed{} final answers. These are the
# canonical demos used in the original MATH paper appendix and lm-eval.

_MATH_SHOTS: List[Tuple[str, str]] = [
    (
        "Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
        "The expressions inside each square root must be non-negative. "
        "Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le "
        "5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which "
        "gives $x<5$. Therefore, the domain of the expression is "
        "$\\boxed{[2,5)}$.",
    ),
    (
        "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find "
        "$\\det (\\mathbf{A} \\mathbf{B}).$",
        "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det "
        "\\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$",
    ),
    (
        "Terrell usually lifts two 20-pound weights 12 times. If he uses two "
        "15-pound weights instead, how many times must Terrell lift them in "
        "order to lift the same total weight?",
        "If Terrell lifts two 20-pound weights 12 times, he lifts a total of "
        "$2\\cdot 12\\cdot20=480$ pounds of weight. If he lifts two 15-pound "
        "weights instead for $n$ times, he will lift a total of $2\\cdot15"
        "\\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can "
        "solve for $n$: $30n=480$, so $n=480/30=\\boxed{16}$.",
    ),
    (
        "If the system of equations $6x-4y=a$ and $6y-9x = b$ has a solution "
        "$(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b}$, "
        "assuming $b$ is nonzero.",
        "If we multiply the first equation by $-\\frac{3}{2}$, we obtain "
        "$6y-9x=-\\frac{3}{2}a$. Since we also know that $6y-9x=b$, we have "
        "$-\\frac{3}{2}a=b$, so $\\frac{a}{b}=\\boxed{-\\frac{2}{3}}$.",
    ),
]


def _format_math(problem: str) -> str:
    parts: List[str] = []
    for q, a in _MATH_SHOTS:
        parts.append(f"Problem: {q}\nSolution: {a}")
    parts.append(f"Problem: {problem}\nSolution:")
    return "\n\n".join(parts)


_MATH_STOPS = ["\nProblem:", "\n\nProblem:"]


# ----------------------------------------------------------------- HumanEval --
# 0-shot per Dream paper. Just the function signature + docstring.

def _format_humaneval(prompt: str) -> str:
    return prompt


# Stop the model from continuing past the function body into another def/class
# or a "if __name__" guard, which is the standard pattern in the human-eval
# reference harness.
_HUMANEVAL_STOPS = [
    "\ndef ",
    "\nclass ",
    "\nif __name__",
    "\nprint(",
    "\n#",
    "\n```",
]


# ---------------------------------------------------------------------- MBPP --
# 4-shot BEGIN/DONE format from the original MBPP paper, also used by
# lm-evaluation-harness. Each demo includes the task, the expected tests, and
# the canonical solution wrapped in [BEGIN]...[DONE].

_MBPP_PREAMBLE = (
    "You are an expert Python programmer, and here is your task: "
)

_MBPP_SHOTS: List[Tuple[str, str, str]] = [
    (
        "Write a function to find the similar elements from the given two "
        "tuple lists.",
        "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\n"
        "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)\n"
        "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == "
        "(13, 14)",
        "def similar_elements(test_tup1, test_tup2):\n"
        "  res = tuple(set(test_tup1) & set(test_tup2))\n"
        "  return (res)",
    ),
    (
        "Write a python function to identify non-prime numbers.",
        "assert is_not_prime(2) == False\n"
        "assert is_not_prime(10) == True\n"
        "assert is_not_prime(35) == True",
        "import math\n"
        "def is_not_prime(n):\n"
        "    result = False\n"
        "    for i in range(2,int(math.sqrt(n)) + 1):\n"
        "        if n % i == 0:\n"
        "            result = True\n"
        "    return result",
    ),
    (
        "Write a function to find the largest integers from a given list of "
        "numbers using heap queue algorithm.",
        "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)"
        "==[85, 75, 65]\n"
        "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)"
        "==[85, 75]\n"
        "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)"
        "==[85, 75, 65, 58, 35]",
        "import heapq as hq\n"
        "def heap_queue_largest(nums,n):\n"
        "  largest_nums = hq.nlargest(n, nums)\n"
        "  return largest_nums",
    ),
    (
        "Write a function to find the maximum total path sum in the given "
        "triangle.",
        "assert max_path_sum([[1, 0, 0], [4, 8, 0], [1, 5, 3]], 2, 2) == 14\n"
        "assert max_path_sum([[13, 0, 0], [7, 4, 0], [2, 4, 6]], 2, 2) == 24\n"
        "assert max_path_sum([[2, 0, 0], [11, 18, 0], [10, 7, 6]], 2, 2) == 31",
        "def max_path_sum(tri, m, n):\n"
        "  for i in range(m-1, -1, -1):\n"
        "    for j in range(i+1):\n"
        "      if (tri[i+1][j] > tri[i+1][j+1]):\n"
        "        tri[i][j] += tri[i+1][j]\n"
        "      else:\n"
        "        tri[i][j] += tri[i+1][j+1]\n"
        "  return tri[0][0]",
    ),
]


def _format_mbpp(prompt: str, test: str) -> str:
    parts: List[str] = []
    for q, tests, code in _MBPP_SHOTS:
        parts.append(
            f"{_MBPP_PREAMBLE}{q} Your code should pass these tests:\n\n"
            f"{tests}\n[BEGIN]\n{code}\n[DONE]"
        )
    parts.append(
        f"{_MBPP_PREAMBLE}{prompt} Your code should pass these tests:\n\n"
        f"{test}\n[BEGIN]\n"
    )
    return "\n\n".join(parts)


_MBPP_STOPS = ["[DONE]", "\nYou are an expert"]


# ------------------------------------------------------------------- IFEval --

def _format_ifeval(prompt: str) -> str:
    return prompt


_IFEVAL_STOPS: List[str] = []


# --------------------------------------------------------------------- main --

def iter_prompts(
    benchmark: str,
    split: str = "train",
    limit: Optional[int] = None,
) -> Iterator[Prompt]:
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
                stop_sequences=list(_GSM8K_STOPS),
            )
    elif bm == "math":
        candidates = [
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
            yield Prompt(
                prompt_id=i,
                benchmark="math",
                text=_format_math(ex["problem"]),
                reference=ex.get("solution"),
                stop_sequences=list(_MATH_STOPS),
            )
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
                stop_sequences=list(_MBPP_STOPS),
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
                stop_sequences=list(_HUMANEVAL_STOPS),
            )
    elif bm == "ifeval":
        ds = load_dataset("HuggingFaceH4/ifeval", split="train")
        for i, ex in enumerate(ds):
            if limit is not None and i >= limit:
                break
            yield Prompt(
                prompt_id=i,
                benchmark="ifeval",
                text=_format_ifeval(ex["prompt"]),
                stop_sequences=list(_IFEVAL_STOPS),
            )
    elif bm in ("owt", "openwebtext"):
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        for i, ex in enumerate(ds):
            if limit is not None and i >= limit:
                break
            text = ex["text"][:512]
            yield Prompt(prompt_id=i, benchmark="owt", text=text)
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")


def collect_prompts(
    benchmark: str,
    split: str = "train",
    limit: Optional[int] = None,
) -> List[Prompt]:
    return list(iter_prompts(benchmark, split=split, limit=limit))
