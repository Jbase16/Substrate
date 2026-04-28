"""
scripts/test_controller_loop.py — Integration test for the full control spine.

The chain under test:

    ScriptedVerifier  →  TierController  →  active ScheduledOp tier
                                                      ↓
                          MLXOpKernel  ←  WeightBank.swap (via kernel.execute)
                                                      ↓
                                                changed logits

If any link is broken, the test reveals which one. The test asserts at
THREE points along the chain so failures localize:

    (1) controller.active_op(op_id).tier_index == expected
        (the controller actually escalated)

    (2) bank.active_tier(op_id) == expected
        (the bank actually swapped — kernel forwarded the tier through)

    (3) logits match the in-place oracle for the new tier
        (the swap installed the correct weights)

Failure modes the assertions distinguish:
    (1) fails alone   — verifier signal not reaching controller
    (1) ok, (2) fails — kernel ignored tier_index, or bank.swap silently failed
    (1,2) ok, (3) fails — bank installed wrong weights (numerical bug)

This is Step 1 of the verifier integration: scripted disagreement values,
no probes. Probe training is Step 2; real verifier is Step 3.

Test scenario:

    Plan: 3 tiers per op
        tier 0 = 4-bit (default, baseline)
        tier 1 = 6-bit (escalation 1)
        tier 2 = fp16  (escalation 2)

    EscalationPolicy:
        disagreement_threshold = 0.15
        consecutive_hits_for_tier_2 = 3
        persistence_tokens = 32 (long enough that demotion doesn't bite
                                 during the test window)
        max_concurrent_escalations = 4

    Driver loop (runs N tokens of a fake forward):
        For each token:
            For each op in plan (in topological order):
                active = controller.active_op(op.op_id)
                hidden = kernel.execute(active, decision=None, hidden=hidden)
                disagreement = scripted_verifier.disagreement(op.op_id, hidden)
                controller.observe(op.op_id, disagreement)
            controller.end_token()

    Script: layer_0.attention gets disagreement=0.5 for the first 3 tokens
            (well above threshold), forcing tier 0 -> tier 1 -> tier 2.
            All other ops always see disagreement=0 (below threshold).

Assertions after token 1 (one hit):
    controller has layer_0.attention at tier 1
    bank has layer_0.attention at tier 1

Assertions after token 3 (three hits):
    controller has layer_0.attention at tier 2
    bank has layer_0.attention at tier 2

Final logits comparison:
    Run forward with controller pinning layer_0.attention at tier 2.
    Compare against an oracle: fresh model with all ops at 4-bit EXCEPT
    layer_0.self_attn left at fp16 (tier 2's precision).
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from typing import Any


def _import_mlx():
    import mlx.core as mx
    import mlx_lm
    return mx, mlx_lm


# ---------------------------------------------------------------------------
# Comparison metrics.
# ---------------------------------------------------------------------------
def cosine_similarity(a, b):
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0 if (na != 0 or nb != 0) else 1.0
    return dot / (na * nb)


def max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def extract_last_logits(logits_array):
    last = logits_array[0, -1, :]
    return [float(v) for v in last.tolist()]


# ---------------------------------------------------------------------------
# A scripted verifier specifically for this test.
#
# substrate.runtime.verifier.ScriptedVerifier exists but its API mutates
# the script via list.pop, which means rerunning the same op_id consumes
# the script. For this test we want predictable per-token behavior with
# multiple op observations per token, so we wrap it in a per-token-step
# generator that the test driver advances.
# ---------------------------------------------------------------------------
class _PerTokenScriptVerifier:
    """
    Scripts disagreement values per (op_id, token_index).

    schedule[op_id] is a list of disagreement values, one per token.
    On each observe() call we look up the current token's value and return it.
    Token index is bumped externally via advance_token().

    Ops not in the schedule always return 0.0 (below any reasonable threshold).
    """

    def __init__(self, schedule: dict[str, list[float]]):
        self._schedule = dict(schedule)
        self._token_idx = 0

    def disagreement(self, op_id: str, hidden) -> float:
        values = self._schedule.get(op_id)
        if values is None:
            return 0.0
        if self._token_idx >= len(values):
            return 0.0
        return values[self._token_idx]

    def advance_token(self) -> None:
        self._token_idx += 1

    def reset(self) -> None:
        self._token_idx = 0


# ---------------------------------------------------------------------------
# Driver loop. Walks one "token" worth of forward, calling the controller
# along the way. NOT the real Executor — direct integration to keep the
# test focused on the spine.
# ---------------------------------------------------------------------------
def run_one_token(
    session, kernel, controller, verifier, plan, tokens, *,
    track_active_tiers: dict[str, list[int]] | None = None,
):
    """
    Runs one forward pass through the kernel with the controller deciding
    each op's tier. Returns final logits.

    track_active_tiers: optional dict where, for each op_id of interest,
                        we append the tier_index used for THIS token.
                        Useful for test assertions that want to know
                        "what tier was actually executed when?"
    """
    mx, _ = _import_mlx()
    session.kernel.reset_caches()
    hidden = session._embed(tokens)

    # The plan's op_bundles are in topological order, which matches the
    # kernel's expected execute order.
    for ob in plan.op_bundles:
        active_op = controller.active_op(ob.op_id)
        if track_active_tiers is not None and ob.op_id in track_active_tiers:
            track_active_tiers[ob.op_id].append(active_op.tier_index)
        hidden = kernel.execute(active_op, decision=None, hidden=hidden)
        # Verifier observes the post-op hidden state. Mean-pooled over
        # seq for parity with calibration's mean-pooled measurement.
        disagreement = verifier.disagreement(ob.op_id, hidden)
        controller.observe(ob.op_id, disagreement)

    hidden = session._final_norm(hidden)
    logits = session._lm_head(hidden)
    mx.eval(logits)

    controller.end_token()
    verifier.advance_token()
    return logits


# ---------------------------------------------------------------------------
# The test.
# ---------------------------------------------------------------------------
def run_test_cl1(model_id: str, prompt: str, verbose: bool) -> int:
    mx, mlx_lm = _import_mlx()

    print("=" * 70)
    print("TEST CL1: ScriptedVerifier → TierController → bank → kernel")
    print("=" * 70)
    print(f"Model:  {model_id}")
    print(f"Prompt: {prompt!r}")
    print()

    # ------------------------------------------------------------------
    # Setup: model, session, plan, store, bank.
    # ------------------------------------------------------------------
    print("Loading model...")
    model, tokenizer = mlx_lm.load(model_id)
    tokens_list = tokenizer.encode(prompt)
    if hasattr(tokens_list, "tolist"):
        tokens_list = tokens_list.tolist()
    tokens = mx.array([tokens_list])

    from substrate.backend.mlx_session import MLXForwardSession
    session = MLXForwardSession(model)
    layers = session.kernel._layers
    print(f"  {len(layers)} layers")
    print()

    # 3-tier plan: tier 0 = 4-bit, tier 1 = 6-bit, tier 2 = fp16.
    # The IR sorts these by precision; in build_test_plan's encoding,
    # tier 0 is 4-bit (lowest bits), tier 2 is fp16 (highest bits).
    print("Building 3-tier plan (4bit → 6bit → fp16)...")
    from scripts.test_weight_bank import build_test_plan, build_op_tier_precisions
    plan = build_test_plan(
        model_id, layers,
        {0: "4bit", 1: "6bit", 2: "fp16_eq"},
    )
    precisions = build_op_tier_precisions(
        plan, {0: "4bit", 1: "6bit", 2: "fp16_eq"},
    )

    # Make the policy explicit so the test is self-documenting and
    # robust against future default changes.
    from substrate.compiler.ir import EscalationPolicy
    policy = EscalationPolicy(
        disagreement_threshold=0.15,
        consecutive_hits_for_tier_2=3,
        persistence_tokens=32,
        max_concurrent_escalations=4,
        enable_demotion=True,
    )
    # Replace the plan's policy. PlanBundle is frozen, so we rebuild it
    # via dataclasses.replace.
    import dataclasses
    plan = dataclasses.replace(plan, escalation_policy=policy)
    print(f"  EscalationPolicy: {policy}")
    print()

    print("Building store and bank...")
    from substrate.backend.ram_weight_store import RAMWeightStore
    from substrate.backend.weight_bank import WeightBank
    ops_with_tiers = {ob.op_id: [0, 1, 2] for ob in plan.op_bundles}
    store = RAMWeightStore(model, layers, ops_with_tiers, precisions)
    bank = WeightBank(model, layers, store, plan)
    session.attach_weight_bank(bank)
    print(f"  Bank ready: all ops at tier 0 (4-bit)")
    print()

    # ------------------------------------------------------------------
    # Setup: controller + scripted verifier.
    # ------------------------------------------------------------------
    print("Building TierController...")
    from substrate.runtime.tier_controller import TierController
    controller = TierController(plan)
    print(f"  controller initialized; pool_size={controller.pool_size_bytes / 1e6:.1f} MB")
    print()

    print("Building scripted verifier...")
    # Script: layer_0.attention sees disagreement=0.5 for tokens 0, 1, 2,
    # then 0.0 forever. Threshold is 0.15, so each of those is a hit.
    # 1st hit -> tier 1. 3rd consecutive hit -> tier 2.
    target_op = "layer_0.attention"
    verifier = _PerTokenScriptVerifier({
        target_op: [0.5, 0.5, 0.5, 0.0, 0.0],
    })
    print(f"  scripted verifier: {target_op} disagrees 3 tokens")
    print()

    # ------------------------------------------------------------------
    # Drive the loop.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Driving controller for 4 tokens, watching escalation...")
    print("=" * 70)

    track = {target_op: []}

    print()
    print("-- Token 0 (first hit, expect tier 1) --")
    logits_t0 = run_one_token(
        session, session.kernel, controller, verifier, plan, tokens,
        track_active_tiers=track,
    )
    print(f"  controller active_op({target_op}).tier_index = "
          f"{controller.active_op(target_op).tier_index}")
    print(f"  bank.active_tier({target_op}) = {bank.active_tier(target_op)}")
    print(f"  tier executed during this token: {track[target_op][-1]}")
    print()

    print("-- Token 1 (second hit) --")
    logits_t1 = run_one_token(
        session, session.kernel, controller, verifier, plan, tokens,
        track_active_tiers=track,
    )
    print(f"  controller active_op({target_op}).tier_index = "
          f"{controller.active_op(target_op).tier_index}")
    print(f"  bank.active_tier({target_op}) = {bank.active_tier(target_op)}")
    print(f"  tier executed during this token: {track[target_op][-1]}")
    print()

    print("-- Token 2 (third hit, expect tier 2 by end) --")
    logits_t2 = run_one_token(
        session, session.kernel, controller, verifier, plan, tokens,
        track_active_tiers=track,
    )
    print(f"  controller active_op({target_op}).tier_index = "
          f"{controller.active_op(target_op).tier_index}")
    print(f"  bank.active_tier({target_op}) = {bank.active_tier(target_op)}")
    print(f"  tier executed during this token: {track[target_op][-1]}")
    print()

    print("-- Token 3 (no hit; tier should stay during persistence) --")
    logits_t3 = run_one_token(
        session, session.kernel, controller, verifier, plan, tokens,
        track_active_tiers=track,
    )
    final_tier = controller.active_op(target_op).tier_index
    print(f"  controller active_op({target_op}).tier_index = {final_tier}")
    print(f"  bank.active_tier({target_op}) = {bank.active_tier(target_op)}")
    print(f"  tier executed during this token: {track[target_op][-1]}")
    print()

    # ------------------------------------------------------------------
    # Build the oracle: all ops at 4-bit EXCEPT layer_0.self_attn at fp16.
    # The final state of the controlled run should match this.
    # ------------------------------------------------------------------
    print("Building oracle: in-place 4-bit everywhere EXCEPT layer_0.self_attn (fp16)...")
    model_oracle, _ = mlx_lm.load(model_id)
    session_oracle = MLXForwardSession(model_oracle)
    layers_oracle = session_oracle.kernel._layers
    from substrate.backend.quantize import quantize_module_in_place
    n_modified = 0
    for layer_id in range(len(layers_oracle)):
        if layer_id == 0:
            n_modified += quantize_module_in_place(layers_oracle[0].mlp, bits=4)
            continue
        n_modified += quantize_module_in_place(layers_oracle[layer_id].self_attn, bits=4)
        n_modified += quantize_module_in_place(layers_oracle[layer_id].mlp, bits=4)
    print(f"  Quantized {n_modified} weight arrays")
    print()

    print("Running oracle forward...")
    session_oracle.reset()
    logits_oracle = session_oracle.forward_via_kernel(tokens)
    last_oracle = extract_last_logits(logits_oracle)
    last_t3 = extract_last_logits(logits_t3)
    print(f"  oracle top-1: {max(range(len(last_oracle)), key=lambda i: last_oracle[i])}")
    print(f"  controlled t3 top-1: {max(range(len(last_t3)), key=lambda i: last_t3[i])}")
    print()

    # ------------------------------------------------------------------
    # Assertions at three points along the chain.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Assertions")
    print("=" * 70)

    failures = []

    # Track[target_op] should reflect the tiers EXECUTED during each token.
    # Token 0: starts at tier 0, kernel runs tier 0. AFTER the verifier
    #          observes (high disagreement), controller escalates to tier 1.
    #          So track[0] should be 0 (the tier USED for token 0).
    # Token 1: starts at tier 1 (after token 0's escalation). Verifier
    #          observes again, controller decides tier_2 path... but only
    #          after consecutive_hits_for_tier_2=3. After 2 hits we are
    #          still at tier 1.
    # Token 2: starts at tier 1, kernel runs tier 1. AFTER the verifier
    #          observes (3rd hit), controller jumps to tier 2.
    # Token 3: starts at tier 2, kernel runs tier 2. No more hits.

    expected_executed = [0, 1, 1, 2]
    print(f"  Track of tiers executed: {track[target_op]}")
    print(f"  Expected:                 {expected_executed}")
    if track[target_op] != expected_executed:
        failures.append(
            f"Tier execution sequence wrong: got {track[target_op]}, "
            f"expected {expected_executed}"
        )

    print(f"\n  After token 3:")
    print(f"    controller tier: {controller.active_op(target_op).tier_index}  (expect 2)")
    print(f"    bank tier:       {bank.active_tier(target_op)}  (expect 2)")

    if controller.active_op(target_op).tier_index != 2:
        failures.append(
            f"Controller tier wrong: got {controller.active_op(target_op).tier_index}, expected 2"
        )
    if bank.active_tier(target_op) != 2:
        failures.append(
            f"Bank tier wrong: got {bank.active_tier(target_op)}, expected 2"
        )

    # Logits comparison: t3 (controller settled at tier 2) vs oracle.
    cos = cosine_similarity(last_t3, last_oracle)
    diff = max_abs_diff(last_t3, last_oracle)
    print(f"\n  Logits compare (t3 vs oracle):")
    print(f"    cosine_similarity: {cos:.10f}")
    print(f"    max_abs_diff:      {diff:.6e}")
    if cos < 0.99999 or diff > 0.01:
        failures.append(
            f"Logits at controlled-tier-2 don't match oracle "
            f"(cosine={cos}, diff={diff})"
        )

    print()
    if not failures:
        print("✓ TEST CL1 PASSED")
        print("  The full control spine works:")
        print("    Verifier signal → Controller decision → Kernel execution")
        print("    → Bank swap → Correct logits")
        return 0
    print("✗ TEST CL1 FAILED")
    for f in failures:
        print(f"  - {f}")
    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    return run_test_cl1(args.model, args.prompt, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
