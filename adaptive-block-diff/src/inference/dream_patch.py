"""Optional patch into the AdaBlock-dLLM fork's Dream generator.

Mirrors llada_patch.py exactly; see that file for behaviour.
"""

from __future__ import annotations

import importlib

from .scheduler import LearnedScheduler


_FORK_MODULE = "third_party.AdaBlock-dLLM.dream.generate_adablock"


def install(scheduler: LearnedScheduler) -> bool:
    try:
        mod = importlib.import_module(_FORK_MODULE)
    except ModuleNotFoundError:
        print(
            "[dream_patch] AdaBlock-dLLM fork not found; falling back to "
            "scheduled_sampler.scheduled_rollout for eval.",
            flush=True,
        )
        return False

    if not hasattr(mod, "set_block_size_rule"):
        print(
            "[dream_patch] fork is present but does not expose "
            "set_block_size_rule; manual edit required (see README).",
            flush=True,
        )
        return False

    def _rule(state):
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
    print("[dream_patch] installed LearnedScheduler into AdaBlock-dLLM fork.", flush=True)
    return True
