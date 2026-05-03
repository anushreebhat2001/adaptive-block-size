"""Optional patch into the AdaBlock-dLLM fork's LLaDA generator.

This is only used when you want the headline benchmark numbers reproduced
against the *exact* AdaBlock-dLLM sampler rather than our scheduled sampler.
It expects the fork to be present at:

    third_party/AdaBlock-dLLM/llada/generate_adablock.py

with a top-level function ``generate(...)`` that consumes a callable
``block_size_rule(state) -> int``. The patch swaps that callable for our
LearnedScheduler.

If the fork is not present, importing this module is a no-op so the rest
of the package still installs.
"""

from __future__ import annotations

import importlib
import os
from typing import Optional

from .scheduler import LearnedScheduler


_FORK_MODULE = "third_party.AdaBlock-dLLM.llada.generate_adablock"


def install(scheduler: LearnedScheduler) -> bool:
    """Patch the AdaBlock-dLLM fork to use our scheduler.

    Returns True on success, False if the fork is not present.
    """
    try:
        mod = importlib.import_module(_FORK_MODULE)
    except ModuleNotFoundError:
        print(
            "[llada_patch] AdaBlock-dLLM fork not found; falling back to "
            "scheduled_sampler.scheduled_rollout for eval.",
            flush=True,
        )
        return False

    if not hasattr(mod, "set_block_size_rule"):
        print(
            "[llada_patch] fork is present but does not expose "
            "set_block_size_rule; manual edit required (see README).",
            flush=True,
        )
        return False

    def _rule(state):
        # state is whatever the fork passes -- in the standard fork it's a
        # dict with the same fields as our SchedulerInput. Adapt here if
        # the fork's payload diverges.
        from .scheduler import SchedulerInput

        si = SchedulerInput(
            block_logits=state["block_logits"],
            block_hidden=state["block_hidden"],
            block_token_ids=state["block_token_ids"],
            next_window_token_ids=state["next_window_token_ids"],
            position=state["position"],
        )
        return scheduler.next_block_size(si)

    mod.set_block_size_rule(_rule)
    print("[llada_patch] installed LearnedScheduler into AdaBlock-dLLM fork.", flush=True)
    return True
