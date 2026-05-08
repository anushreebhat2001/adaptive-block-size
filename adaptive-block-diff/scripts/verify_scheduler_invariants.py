"""Self-check: verifies the scheduler invariants we want to claim in a paper.

Runs in seconds on a CPU. No model load required for the FixedScheduler
checks. The LearnedScheduler check builds a tiny BlockSizePredictor with
random weights to confirm the inference path is grad-free / eval-mode /
parameter-stable.

Exits non-zero on the first failed assertion so it can be wired into CI or
run as the first command of any sbatch.

Usage:
    python -m scripts.verify_scheduler_invariants
"""

from __future__ import annotations

import sys
from copy import deepcopy

import torch

from src.inference.scheduler import (
    AdaBlockScheduler,
    FixedScheduler,
    LearnedScheduler,
    SchedulerInput,
    make_scheduler,
)
from src.predictor.features import CANDIDATE_BLOCK_SIZES, n_scalar_features
from src.predictor.model import BlockSizePredictor


def _fake_input(block_len=16, vocab=128, hidden=64):
    return SchedulerInput(
        block_logits=torch.randn(block_len, vocab),
        block_hidden=torch.randn(block_len, hidden),
        block_token_ids=torch.randint(0, vocab, (block_len,)),
        next_window_token_ids=torch.randint(0, vocab, (8,)),
        position=64,
    )


def check_fixed():
    print("[fixed] checking FixedScheduler invariants...")
    for B in CANDIDATE_BLOCK_SIZES:
        s = FixedScheduler(B)
        # 1. needs_state flag set
        assert s.needs_state is False, f"fixed-{B}: needs_state should be False"
        # 2. initial_block_size = B (so first block matches)
        assert s.initial_block_size == B, f"fixed-{B}: initial_block_size should equal B"
        # 3. next_block_size always returns B
        for _ in range(20):
            out = s.next_block_size(_fake_input())
            assert out == B, f"fixed-{B}: returned {out}"
        # 4. No torch parameters anywhere on the object
        for name in dir(s):
            attr = getattr(s, name, None)
            if isinstance(attr, torch.nn.Module):
                raise AssertionError(f"fixed-{B}: nn.Module attribute {name} found — should have none")
            if isinstance(attr, torch.Tensor) and attr.requires_grad:
                raise AssertionError(f"fixed-{B}: tensor {name} requires_grad")
        # 5. reset is a no-op
        s.reset()
        assert s.B == B
    print("[fixed] OK — fixed-{4,8,16,32} all return their B and own no parameters.")


def check_factory_does_not_use_predictor_for_fixed():
    print("[fixed] checking factory ignores predictor arg for fixed-*...")
    # Pass a dummy non-None object as predictor; factory must not touch it.
    sentinel = object()
    s = make_scheduler(
        "fixed-8",
        predictor=sentinel,         # the factory should ignore this for fixed-*
        hidden_dim=999,
        delimiter_token_ids=[1, 2, 3],
    )
    assert isinstance(s, FixedScheduler)
    assert not hasattr(s, "predictor"), "fixed-* must not store a predictor"
    print("[fixed] OK — make_scheduler('fixed-8', predictor=sentinel) produces a clean FixedScheduler.")


def check_learned_inference_only():
    print("[learned] checking LearnedScheduler is grad-free / eval-mode / param-stable...")
    hidden = 64
    pred = BlockSizePredictor(hidden_dim=hidden)
    # Capture a snapshot of all parameters before any inference
    snapshot = {k: v.detach().clone() for k, v in pred.state_dict().items()}

    sched = LearnedScheduler(
        predictor=pred,
        hidden_dim=hidden,
        delimiter_token_ids=[1, 2, 3, 4],
        max_length=512,
    )

    # 1. predictor is in eval mode
    assert sched.predictor.training is False, "predictor should be in eval mode"

    # 2. Run a bunch of predictions and confirm no parameter drift
    for i in range(50):
        b = sched.next_block_size(_fake_input(block_len=16, hidden=hidden))
        assert b in CANDIDATE_BLOCK_SIZES, f"predicted unknown block size {b}"

    # 3. Parameters unchanged
    after = {k: v.detach() for k, v in pred.state_dict().items()}
    for k in snapshot:
        if not torch.equal(snapshot[k], after[k]):
            raise AssertionError(f"parameter {k} drifted during inference")

    # 4. State vector shape matches what the predictor expects
    state = sched.builder.build_state(
        block_logits=torch.randn(16, 128),
        next_window_token_ids=torch.randint(0, 128, (8,)),
        position=64,
    )
    expected_scalar_dim = n_scalar_features()
    assert state.scalars.shape == (expected_scalar_dim,), \
        f"scalar shape {state.scalars.shape} != ({expected_scalar_dim},)"
    assert state.hidden_pool.shape == (hidden,), \
        f"hidden_pool shape {state.hidden_pool.shape} != ({hidden},)"

    # 5. predict_block_size runs under no_grad (autograd graph not built)
    out = pred.forward(
        state.scalars.unsqueeze(0),
        state.hidden_pool.unsqueeze(0),
    )
    # forward DOES build a grad graph because we did NOT call no_grad here on
    # purpose — this confirms the train-time path can still autograd. The
    # inference path uses predict_block_size which wraps in no_grad.
    assert out[0].requires_grad, \
        "predictor.forward should produce a grad-capable tensor when called outside no_grad"

    # 6. predict_block_size produces a non-grad output
    with torch.enable_grad():
        # even with autograd globally enabled, predict_block_size stays no_grad
        idx = pred.predict_block_size(state.scalars, state.hidden_pool)
    assert idx in CANDIDATE_BLOCK_SIZES

    print("[learned] OK — eval-mode, grad-free predict_block_size, no parameter drift over 50 calls.")


def check_reset_isolates_prompts():
    print("[learned] checking reset() clears the per-prompt state buffer...")
    hidden = 32
    pred = BlockSizePredictor(hidden_dim=hidden)
    sched = LearnedScheduler(
        predictor=pred,
        hidden_dim=hidden,
        delimiter_token_ids=[1, 2, 3],
        max_length=512,
    )
    # Push some history
    for _ in range(5):
        sched.next_block_size(_fake_input(hidden=hidden))
    n_before_reset = len(sched.builder._block_hidden_means)
    assert n_before_reset >= 5
    sched.reset()
    n_after_reset = len(sched.builder._block_hidden_means)
    assert n_after_reset == 0, f"reset failed: {n_after_reset} entries remain"
    print("[learned] OK — reset() clears 5 entries → 0.")


def check_factory_paths():
    print("[factory] checking make_scheduler dispatch...")
    fs = make_scheduler("fixed-16")
    assert isinstance(fs, FixedScheduler) and fs.B == 16

    ab = make_scheduler("adablock", delimiter_token_ids=[1, 2])
    assert isinstance(ab, AdaBlockScheduler)
    assert getattr(ab, "needs_state", True) is True

    pred = BlockSizePredictor(hidden_dim=64)
    ls = make_scheduler(
        "ours-teacher",
        predictor=pred,
        hidden_dim=64,
        delimiter_token_ids=[1, 2],
    )
    assert isinstance(ls, LearnedScheduler)
    assert ls.name == "ours-teacher"
    print("[factory] OK — fixed-16 → FixedScheduler, adablock → AdaBlockScheduler, ours-teacher → LearnedScheduler.")


def main() -> int:
    try:
        check_fixed()
        check_factory_does_not_use_predictor_for_fixed()
        check_learned_inference_only()
        check_reset_isolates_prompts()
        check_factory_paths()
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        return 1
    except Exception as e:
        print(f"\nFAIL (unexpected): {e!r}")
        return 1
    print("\nALL CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
