"""
scripts/test_real_verifier_loop.py — Real LinearProbeVerifier integration test.

CL1 (test_controller_loop.py) proved the spine works with a ScriptedVerifier
that emitted hand-picked disagreement values. This test replaces the script
with real probes trained on Qwen2.5-1.5B calibration data and asks:

    Do the trained probes produce enough runtime signal to drive the
    already-proven control loop on a non-calibration prompt?

Test target chain (same as CL1, real verifier):

    LinearProbeVerifier (loaded from probes.json)
        ↓ disagreement(op_id, hidden) — real predictions
    TierController.observe(op_id, value)
        ↓ controller may escalate based on policy threshold
    controller.active_op(op_id) returns ScheduledOp with new tier_index
        ↓
    MLXOpKernel.execute reads tier_index, calls bank.swap()
        ↓
    Bank installs higher-tier weights, forward runs against them
        ↓
    Logits change

Prompt: a TECHNICAL prompt outside the calibration corpus (Pride and
Prejudice). Calibration corpus is literary 19th-century English; this
prompt is technical 21st-century English. Out-of-distribution by design,
which is the honest stress test for the probes.

Two-phase comparison:
    Phase A (baseline): run with threshold=infinity → no escalation
                        possible, every op stays at tier 0 (4-bit).
                        Captures logits_baseline.
    Phase B (escalation): run with policy threshold=0.005 → controller
                          can escalate ops where disagreement crosses
                          threshold. Captures logits_escalated.

Modest assertions (hard):
    1. max observed disagreement > 0 (probes have signal)
    2. at least one op escalated (signal crossed threshold)
    3. some bank.active_tier(op) > 0 (bank received and applied the swap)
    4. logits_escalated != logits_baseline (escalations changed the math)

Diagnostic prints (soft, no assertions):
    - per-op disagreement distribution (min/median/max)
    - which ops escalated, to which tier
    - how often each op escalated across the 4 tokens

This test does NOT assert "verifier picks the right ops" or "answer
quality improves." Those questions need their own evaluation framework.
This test asserts: trained verifier produces signal that drives loop.

Threshold rationale (0.005):
    Per held-out eval on the calibration data:
        median label_mean across ops: 0.0013
        max label_mean (layer_0.attention): 0.0256
        median MAE: 9.5e-5
    0.005 ≈ 4× median label_mean. High enough to filter most ops (which
    have small label_mean and would never see 0.005 if probes are
    well-calibrated), low enough that high-divergence ops will trigger.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Lazy MLX import.
# ---------------------------------------------------------------------------
def _import_mlx():
    import mlx.core as mx
    import mlx_lm
    return mx, mlx_lm


# ---------------------------------------------------------------------------
# Comparison helpers.
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
# Mean-pooled hidden state for the verifier.
#
# Calibration captures pooled vectors (mean over seq axis) per op output.
# The verifier was trained on those. At runtime we have the post-op hidden
# state of shape [1, seq, hidden_dim]; we mean-pool over seq before feeding
# it to the probe so the runtime distribution matches the training one.
# ---------------------------------------------------------------------------
def mean_pool_hidden(hidden) -> list[float]:
    """
    Mean-pool a hidden state across the seq axis.
    hidden: mx.array of shape [batch, seq, hidden_dim]
    Returns: list of floats, length hidden_dim
    """
    import mlx.core as mx
    pooled = mx.mean(hidden, axis=1)  # [batch, hidden_dim]
    pooled = pooled[0]  # batch=1; pick the first
    mx.eval(pooled)
    return [float(v) for v in pooled.tolist()]


# ---------------------------------------------------------------------------
# Driver: walk one token through kernel + controller + verifier.
# ---------------------------------------------------------------------------
def run_one_token(
    session, kernel, controller, verifier, plan, tokens, *,
    threshold_override: float | None = None,
    track_per_op_disagreement: dict[str, list[float]] | None = None,
    track_active_tiers: dict[str, list[int]] | None = None,
):
    """
    Runs one forward pass with the controller deciding each op's tier
    and the verifier observing post-op hidden states.

    threshold_override: if set, overrides the controller's policy
        disagreement_threshold for THIS observation only. Used to run
        a 'baseline' phase where threshold=infinity ensures no
        escalation regardless of what the verifier reports.

    Returns the final logits (mlx array).
    """
    import mlx.core as mx

    session.kernel.reset_caches()
    hidden = session._embed(tokens)

    for ob in plan.op_bundles:
        active_op = controller.active_op(ob.op_id)
        if track_active_tiers is not None and ob.op_id in track_active_tiers:
            track_active_tiers[ob.op_id].append(active_op.tier_index)

        # Execute the op via kernel; bank swaps if needed.
        hidden = kernel.execute(active_op, decision=None, hidden=hidden)

        # Verifier reads the post-op hidden, predicts disagreement.
        # Mean-pool to match calibration's training distribution.
        pooled = mean_pool_hidden(hidden)
        disagreement = verifier.disagreement(ob.op_id, pooled)
        if track_per_op_disagreement is not None:
            track_per_op_disagreement[ob.op_id].append(disagreement)

        # Optionally suppress escalation by feeding 0 to the controller
        # when in baseline mode. This lets us share the same forward
        # path between phases.
        observed = disagreement if threshold_override is None else 0.0
        controller.observe(ob.op_id, observed)

    hidden = session._final_norm(hidden)
    logits = session._lm_head(hidden)
    mx.eval(logits)
    controller.end_token()
    return logits


# ---------------------------------------------------------------------------
# Test orchestration.
# ---------------------------------------------------------------------------
def run_test_rv1(
    model_id: str,
    probes_path: str,
    prompt: str,
    threshold: float,
    num_tokens: int,
    runtime_stats: bool,
    warmup: int,
    verbose: bool,
) -> int:
    mx, mlx_lm = _import_mlx()

    print("=" * 70)
    print("TEST RV1: real LinearProbeVerifier → controller → bank → kernel")
    print("=" * 70)
    print(f"Model:    {model_id}")
    print(f"Probes:   {probes_path}")
    print(f"Prompt:   {prompt!r}")
    print(f"Threshold: {threshold}  (controller policy)")
    print(f"Tokens:   {num_tokens}")
    print(f"Verifier: runtime_stats={runtime_stats}, warmup={warmup}")
    print()

    # ------------------------------------------------------------------
    # Load model.
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
    print(f"  {len(layers)} layers, {len(tokens_list)} prompt tokens")
    print()

    # ------------------------------------------------------------------
    # Build plan, store, bank.
    # ------------------------------------------------------------------
    print("Building 3-tier plan (4bit -> 6bit -> fp16)...")
    from scripts.test_weight_bank import build_test_plan, build_op_tier_precisions
    plan = build_test_plan(
        model_id, layers,
        {0: "4bit", 1: "6bit", 2: "fp16_eq"},
    )
    precisions = build_op_tier_precisions(
        plan, {0: "4bit", 1: "6bit", 2: "fp16_eq"},
    )

    from substrate.compiler.ir import EscalationPolicy
    policy = EscalationPolicy(
        disagreement_threshold=threshold,
        consecutive_hits_for_tier_2=3,
        persistence_tokens=32,
        max_concurrent_escalations=4,
        enable_demotion=True,
    )
    plan = dataclasses.replace(plan, escalation_policy=policy)
    print(f"  policy: threshold={threshold}, tier2_hits=3, persistence=32, max_concurrent=4")
    print()

    print("Building store + bank (3 tiers per op for {} ops)...".format(len(plan.op_bundles)))
    from substrate.backend.ram_weight_store import RAMWeightStore
    from substrate.backend.weight_bank import WeightBank
    ops_with_tiers = {ob.op_id: [0, 1, 2] for ob in plan.op_bundles}
    store = RAMWeightStore(model, layers, ops_with_tiers, precisions)
    bank = WeightBank(model, layers, store, plan)
    session.attach_weight_bank(bank)
    print(f"  bank ready: all ops at tier 0 (4-bit)")
    print()

    # ------------------------------------------------------------------
    # Build the real verifier from saved probes.
    # ------------------------------------------------------------------
    print(f"Loading LinearProbeVerifier from {probes_path}...")
    from substrate.runtime.verifier import LinearProbeVerifier
    verifier = LinearProbeVerifier(
        probes_path,
        runtime_stats=runtime_stats,
        warmup=warmup,
    )
    print(f"  verifier ready  (runtime_stats={runtime_stats}, warmup={warmup})")
    print()

    # ------------------------------------------------------------------
    # PHASE A: baseline — threshold suppressed, no escalation possible.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PHASE A: baseline (no escalation; all ops forced to tier 0)")
    print("=" * 70)

    from substrate.runtime.tier_controller import TierController
    controller_a = TierController(plan)

    # Track tier executed per op per token (baseline should always be 0).
    track_a: dict[str, list[int]] = defaultdict(list)
    # Track verifier output even though we suppress its effect — useful
    # to see what the probes are saying.
    disagreements_a: dict[str, list[float]] = defaultdict(list)

    for t in range(num_tokens):
        logits_a = run_one_token(
            session, session.kernel, controller_a, verifier, plan, tokens,
            threshold_override=0.0,  # forces controller.observe(0)
            track_per_op_disagreement=disagreements_a,
            track_active_tiers={op.op_id: track_a[op.op_id] for op in plan.op_bundles},
        )

    last_a = extract_last_logits(logits_a)
    top1_a = max(range(len(last_a)), key=lambda i: last_a[i])
    print(f"  top-1: {top1_a} ({tokenizer.decode([top1_a])!r})")

    # Verify baseline ran at tier 0 for every op (sanity check).
    bad_baseline = [
        op_id for op_id, tiers in track_a.items()
        if any(t != 0 for t in tiers)
    ]
    assert not bad_baseline, (
        f"PHASE A baseline should have all ops at tier 0, but these escalated: "
        f"{bad_baseline}. The threshold_override didn't work."
    )
    print(f"  ✓ all {len(track_a)} ops stayed at tier 0 across {num_tokens} tokens")
    print()

    # Print observed disagreement distribution from PHASE A. Even though
    # we suppressed the controller, the verifier still ran and produced
    # values; this tells us what the real probes report on this prompt.
    all_vals = [v for vals in disagreements_a.values() for v in vals]
    if all_vals:
        all_vals_sorted = sorted(all_vals)
        n = len(all_vals_sorted)
        median = all_vals_sorted[n // 2]
        p90 = all_vals_sorted[int(n * 0.9)]
        max_v = all_vals_sorted[-1]
        max_op = max(disagreements_a.items(),
                     key=lambda kv: max(kv[1]) if kv[1] else 0)
        print(f"  observed disagreement distribution (across {n} op-token pairs):")
        print(f"    min:    {min(all_vals):.6f}")
        print(f"    median: {median:.6f}")
        print(f"    p90:    {p90:.6f}")
        print(f"    max:    {max_v:.6f}  (op: {max_op[0]})")
        print(f"    # over threshold ({threshold}): "
              f"{sum(1 for v in all_vals if v > threshold)}")
    print()

    # ------------------------------------------------------------------
    # PHASE B: real verifier drives the controller.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PHASE B: real verifier drives controller")
    print("=" * 70)

    # Fresh controller and bank state for clean comparison.
    controller_b = TierController(plan)
    bank.reset_to_tier_0()
    verifier.reset()

    track_b: dict[str, list[int]] = defaultdict(list)
    disagreements_b: dict[str, list[float]] = defaultdict(list)

    for t in range(num_tokens):
        print(f"  -- token {t} --")
        logits_b = run_one_token(
            session, session.kernel, controller_b, verifier, plan, tokens,
            threshold_override=None,  # controller observes for real
            track_per_op_disagreement=disagreements_b,
            track_active_tiers={op.op_id: track_b[op.op_id] for op in plan.op_bundles},
        )
        # Quick status: how many ops have escalated by end of this token?
        escalated_at_end = [
            op_id for op_id in track_b
            if controller_b.active_op(op_id).tier_index > 0
        ]
        print(f"     escalated ops at end of token: {len(escalated_at_end)}")
        if escalated_at_end and verbose:
            print(f"     {escalated_at_end[:5]}{'...' if len(escalated_at_end) > 5 else ''}")

    last_b = extract_last_logits(logits_b)
    top1_b = max(range(len(last_b)), key=lambda i: last_b[i])
    print(f"  top-1: {top1_b} ({tokenizer.decode([top1_b])!r})")
    print()

    # ------------------------------------------------------------------
    # Diagnostics.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Per-op summary (PHASE B)")
    print("=" * 70)

    # Find ops where any tier > 0 was executed.
    escalated_ops = []
    for op_id, tiers in track_b.items():
        if any(t > 0 for t in tiers):
            max_tier = max(tiers)
            n_escalated_tokens = sum(1 for t in tiers if t > 0)
            disagreement_max = max(disagreements_b.get(op_id, [0.0]))
            escalated_ops.append({
                "op_id": op_id,
                "max_tier": max_tier,
                "n_escalated_tokens": n_escalated_tokens,
                "max_disagreement": disagreement_max,
                "tier_sequence": tiers,
            })
    escalated_ops.sort(key=lambda r: r["max_disagreement"], reverse=True)

    print(f"  escalated ops: {len(escalated_ops)} of {len(track_b)} total")
    if escalated_ops:
        print(f"  top escalations (by max disagreement):")
        for row in escalated_ops[:10]:
            print(
                f"    {row['op_id']:<28} max_tier={row['max_tier']}  "
                f"escalated {row['n_escalated_tokens']}/{num_tokens} tokens  "
                f"max_disagreement={row['max_disagreement']:.6f}  "
                f"tiers={row['tier_sequence']}"
            )
    print()

    # Final bank state.
    final_bank_state = bank.active_state()
    bank_promoted_ops = [
        op_id for op_id, tier in final_bank_state.items() if tier > 0
    ]
    print(f"  bank.active_state at end of PHASE B: "
          f"{len(bank_promoted_ops)} ops at tier > 0")
    if bank_promoted_ops:
        # Show distribution of bank tiers.
        tier_counts: dict[int, int] = defaultdict(int)
        for op_id, tier in final_bank_state.items():
            tier_counts[tier] += 1
        print(f"  tier distribution: {dict(sorted(tier_counts.items()))}")
    print()

    # ------------------------------------------------------------------
    # Logits comparison.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Logits A vs B")
    print("=" * 70)
    cos = cosine_similarity(last_a, last_b)
    diff = max_abs_diff(last_a, last_b)
    print(f"  cosine_similarity: {cos:.10f}")
    print(f"  max_abs_diff:      {diff:.6e}")
    print()

    # ------------------------------------------------------------------
    # Hard assertions.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Assertions")
    print("=" * 70)

    failures = []

    # 1. Probes have signal.
    max_disagreement_b = (
        max(max(v) for v in disagreements_b.values() if v)
        if any(disagreements_b.values()) else 0.0
    )
    print(f"  (1) max observed disagreement: {max_disagreement_b:.6f}")
    if max_disagreement_b <= 0:
        failures.append(
            "max disagreement is 0; probes produced no signal at all"
        )

    # 2. At least one op escalated (anywhere across all tokens).
    n_escalated_op_tokens = sum(
        1 for tiers in track_b.values() for t in tiers if t > 0
    )
    print(f"  (2) escalated op-token events: {n_escalated_op_tokens}")
    if n_escalated_op_tokens == 0:
        failures.append(
            "no ops escalated across any token; controller never crossed threshold "
            "(consider lowering threshold, or probes are quieter than expected)"
        )

    # 3. Bank shows at least one op at tier > 0.
    print(f"  (3) bank ops at tier > 0 at end: {len(bank_promoted_ops)}")
    if not bank_promoted_ops and n_escalated_op_tokens > 0:
        failures.append(
            "controller escalated but bank never moved; the kernel→bank "
            "wiring isn't propagating tier_index"
        )

    # 4. Logits A != B (escalation produced different math).
    diff_threshold = 0.001
    print(f"  (4) logits A vs B max_abs_diff: {diff:.6e}  (need > {diff_threshold})")
    if n_escalated_op_tokens > 0 and diff < diff_threshold:
        failures.append(
            f"escalations happened ({n_escalated_op_tokens} op-token events) "
            f"but logits didn't change ({diff:.2e} < {diff_threshold:.0e}). "
            f"Bank installed weights, but the math didn't change?"
        )

    # If no escalations at all, we still want assertion 4 to be informative.
    # The test fails on (2) anyway.

    print()
    if not failures:
        print("✓ TEST RV1 PASSED")
        print(f"  trained verifier produced runtime signal sufficient to drive")
        print(f"  the control loop. {n_escalated_op_tokens} op-token escalations "
              f"caused {diff:.4e} max-logit-divergence vs baseline.")
        return 0
    print("✗ TEST RV1 FAILED")
    for f in failures:
        print(f"  - {f}")
    return 1


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default="mlx-community/Qwen2.5-1.5B-Instruct-bf16",
    )
    p.add_argument(
        "--probes", required=True,
        help="Path to probes.json from train_probe.py.",
    )
    p.add_argument(
        "--prompt",
        default="Explain why SSD streaming is the bottleneck for local LLM inference.",
        help="Prompt to use. Default is intentionally OUT-OF-DOMAIN vs the calibration corpus.",
    )
    p.add_argument(
        "--threshold", type=float, default=0.005,
        help="Controller policy disagreement threshold. Default 0.005 "
             "(roughly 4× median label_mean from training).",
    )
    p.add_argument(
        "--num-tokens", type=int, default=4,
        help="Number of forward passes to drive (analogous to tokens generated).",
    )
    p.add_argument(
        "--no-runtime-stats", action="store_true",
        help=(
            "Disable runtime renormalization; use the saved training-time "
            "feature_mean/feature_std for normalization. This is the "
            "naive behavior — only correct when runtime distribution "
            "matches training distribution."
        ),
    )
    p.add_argument(
        "--warmup", type=int, default=2,
        help=(
            "Number of observations per op to accumulate before freezing "
            "runtime stats and emitting real disagreement output. Default 2 "
            "(small for tests; production likely 8+)."
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    return run_test_rv1(
        model_id=args.model,
        probes_path=args.probes,
        prompt=args.prompt,
        threshold=args.threshold,
        num_tokens=args.num_tokens,
        runtime_stats=not args.no_runtime_stats,
        warmup=args.warmup,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
