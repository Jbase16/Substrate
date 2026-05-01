"""
scripts/test_autoregressive_verifier.py — Real generation with verifier in the loop.

RV1 ran a re-prefill loop: same prompt, same hidden states, no real
token variance. That test proved the control-loop wiring but couldn't
prove that runtime stats and probe ranking actually work in practice.

This test runs autoregressive generation. Each generated token causes a
new forward pass through the kernel with the verifier observing real,
distinct hidden states. The controller's runtime stats see actual
variance; the probe sees actual prompt-induced perturbation; the
controller can escalate AND demote based on patterns over many tokens.

Two-phase comparison:

    Phase A (baseline): generate N tokens with all ops at tier 0 (4-bit).
                        No verifier, no escalation. This is what the
                        model produces under "fixed low precision" —
                        the failure baseline we want the verifier to
                        notice and fix.

    Phase B (adaptive): generate N tokens with verifier + controller
                        + bank. Verifier observes hidden states per op
                        per token; controller decides escalations; bank
                        installs higher-precision weights as needed.

For each phase, we capture:
    - generated tokens (and decoded text)
    - per-token, per-op disagreement values from the verifier
    - tier-of-execution time series per op
    - bank state at the end
    - logits at last step

Hard assertions:

    (1) Both phases generate max_new_tokens without erroring. Cache and
        controller state survive a real generation loop.
    (2) Verifier produces nonzero, non-constant disagreement after warmup.
    (3) Some ops escalate. Some ops do NOT escalate. Discrimination is
        the whole point: if EVERY op goes to tier 2, the verifier is a
        panic button. If ZERO ops escalate, the verifier is a decorative
        light.
    (4) Bank state at the end shows a mix of tiers: at least one op > 0
        and at least one op == 0.
    (5) Phase A and Phase B generate different token sequences (or at
        least have different last-step logits) — adaptive precision
        actually changed model behavior.

Soft diagnostics (printed, not asserted):
    - distribution of disagreement values per token over the generation
    - which ops escalated and to what tier
    - tier transitions over time (escalation/demotion patterns)
    - generated text for both phases

Tunables:
    --threshold: disagreement threshold (default 0.5; matches the post-fix
                 verifier where logit clipping bounds output to ~[0, 1])
    --warmup: probe runtime-stats warmup (default 4)
    --max-new-tokens: how many tokens to generate (default 24)
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import statistics
import sys
from collections import defaultdict
from typing import Any


def _import_mlx():
    import mlx.core as mx
    import mlx_lm
    return mx, mlx_lm


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def mean_pool_hidden(hidden) -> list[float]:
    """Mean-pool [1, seq, hidden_dim] -> list of hidden_dim floats."""
    import mlx.core as mx
    pooled = mx.mean(hidden, axis=1)
    pooled = pooled[0]
    mx.eval(pooled)
    return [float(v) for v in pooled.tolist()]


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


def extract_logits(arr) -> list[float]:
    return [float(v) for v in arr.tolist()]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


# ---------------------------------------------------------------------------
# The test.
# ---------------------------------------------------------------------------
def run_test(
    model_id: str,
    probes_path: str,
    prompt: str,
    threshold: float,
    warmup: int,
    max_new_tokens: int,
    runtime_stats: bool,
    verbose: bool,
) -> int:
    mx, mlx_lm = _import_mlx()

    print("=" * 70)
    print("AUTOREGRESSIVE VERIFIER LOOP TEST")
    print("=" * 70)
    print(f"Model:         {model_id}")
    print(f"Probes:        {probes_path}")
    print(f"Prompt:        {prompt!r}")
    print(f"Threshold:     {threshold}")
    print(f"Warmup:        {warmup}")
    print(f"Max new tok:   {max_new_tokens}")
    print(f"Runtime stats: {runtime_stats}")
    print()

    # ------------------------------------------------------------------
    # Setup.
    # ------------------------------------------------------------------
    print("Loading model...")
    model, tokenizer = mlx_lm.load(model_id)
    toks = tokenizer.encode(prompt)
    if hasattr(toks, "tolist"):
        toks = toks.tolist()
    tokens = mx.array([toks])
    print(f"  prompt tokens: {len(toks)}")

    from substrate.backend.mlx_session import MLXForwardSession
    session = MLXForwardSession(model)
    layers = session.kernel._layers
    print(f"  {len(layers)} layers")
    print()

    print("Building 3-tier plan (4bit -> 6bit -> fp16)...")
    from substrate.bench import build_test_plan, build_op_tier_precisions
    plan = build_test_plan(model_id, layers, {0: "4bit", 1: "6bit", 2: "fp16_eq"})
    precisions = build_op_tier_precisions(plan, {0: "4bit", 1: "6bit", 2: "fp16_eq"})

    from substrate.compiler.ir import EscalationPolicy
    policy = EscalationPolicy(
        disagreement_threshold=threshold,
        consecutive_hits_for_tier_2=3,
        persistence_tokens=8,        # Allow demotion within reasonable test horizon.
        max_concurrent_escalations=8,
        enable_demotion=True,
    )
    plan = dataclasses.replace(plan, escalation_policy=policy)
    print(f"  policy: threshold={threshold}, persistence={policy.persistence_tokens}, "
          f"max_concurrent={policy.max_concurrent_escalations}, demotion={policy.enable_demotion}")
    print()

    print("Building store + bank...")
    from substrate.backend.ram_weight_store import RAMWeightStore
    from substrate.backend.weight_bank import WeightBank
    ops_with_tiers = {ob.op_id: [0, 1, 2] for ob in plan.op_bundles}
    store = RAMWeightStore(model, layers, ops_with_tiers, precisions)
    bank = WeightBank(model, layers, store, plan)
    session.attach_weight_bank(bank)
    print(f"  bank ready, all ops at tier 0")
    print()

    # ------------------------------------------------------------------
    # PHASE A: baseline generation, no verifier, no escalation.
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"PHASE A: baseline ({max_new_tokens} tokens at tier 0 = 4-bit)")
    print("=" * 70)

    bank.reset_to_tier_0()
    result_a = session.generate_via_kernel(
        tokens,
        max_new_tokens=max_new_tokens,
        sample_strategy="argmax",
    )
    text_a = tokenizer.decode(result_a["tokens"])
    print(f"  generated tokens: {result_a['tokens']}")
    print(f"  text: {text_a!r}")
    print(f"  bank state: all ops at tier 0? "
          f"{all(t == 0 for t in bank.active_state().values())}")
    final_logits_a = extract_logits(result_a["final_logits"])
    print()

    # ------------------------------------------------------------------
    # PHASE B: adaptive generation, verifier observes, controller drives.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PHASE B: adaptive (verifier + controller + bank)")
    print("=" * 70)

    bank.reset_to_tier_0()
    from substrate.runtime.tier_controller import TierController
    from substrate.runtime.verifier import LinearProbeVerifier
    controller = TierController(plan)
    verifier = LinearProbeVerifier(
        probes_path,
        runtime_stats=runtime_stats,
        warmup=warmup,
    )

    # Per-op time series. Captured via on_op_complete + on_token hooks.
    disagreements: dict[str, list[float]] = defaultdict(list)
    tier_at_exec: dict[str, list[int]] = defaultdict(list)
    tokens_seen: list[int] = []
    # token_index_per_op_log: index of the generation step at which each
    # disagreement value was observed. The same value is recorded for all
    # ops within a single forward pass; we track it once per pass below.
    current_step = [0]  # mutable container so closures can update it
    current_phase = ["prefill"]  # 'prefill' or 'gen'

    def on_op_complete(op, hidden):
        """Verifier observes post-op hidden state, feeds controller."""
        pooled = mean_pool_hidden(hidden)
        d = verifier.disagreement(op.op_id, pooled)
        disagreements[op.op_id].append(d)
        tier_at_exec[op.op_id].append(op.tier_index)
        controller.observe(op.op_id, d)

    def on_token(token_id, logits, step):
        tokens_seen.append(token_id)
        current_step[0] = step
        current_phase[0] = "gen"

    result_b = session.generate_via_kernel(
        tokens,
        max_new_tokens=max_new_tokens,
        controller=controller,
        plan=plan,
        on_op_complete=on_op_complete,
        on_token=on_token,
        sample_strategy="argmax",
    )
    text_b = tokenizer.decode(result_b["tokens"])
    final_logits_b = extract_logits(result_b["final_logits"])
    print(f"  generated tokens: {result_b['tokens']}")
    print(f"  text: {text_b!r}")
    print()

    # ------------------------------------------------------------------
    # Diagnostics.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Per-op disagreement distribution (over all forward passes)")
    print("=" * 70)
    all_obs = [d for vals in disagreements.values() for d in vals]
    if all_obs:
        # Filter out the warmup-zero values to see the real signal
        # distribution. Warmup outputs 0.0 by design.
        nonzero = [d for d in all_obs if d > 0]
        print(f"  total observations: {len(all_obs)}")
        print(f"  observations during warmup (=0): {len(all_obs) - len(nonzero)}")
        print(f"  post-warmup observations:        {len(nonzero)}")
        if nonzero:
            print(f"    min:    {min(nonzero):.4f}")
            print(f"    p10:    {percentile(nonzero, 0.10):.4f}")
            print(f"    median: {percentile(nonzero, 0.50):.4f}")
            print(f"    p90:    {percentile(nonzero, 0.90):.4f}")
            print(f"    max:    {max(nonzero):.4f}")
            stdev = statistics.stdev(nonzero) if len(nonzero) > 1 else 0.0
            print(f"    stdev:  {stdev:.4f}")
            print(f"    over threshold ({threshold}): {sum(1 for d in nonzero if d > threshold)}/{len(nonzero)}")
    print()

    print("=" * 70)
    print("Per-op behavior summary")
    print("=" * 70)
    op_summary = []
    for op_id, tiers in tier_at_exec.items():
        max_tier = max(tiers) if tiers else 0
        any_escalated = any(t > 0 for t in tiers)
        max_d = max(disagreements.get(op_id, [0.0]))
        n_at_t2 = sum(1 for t in tiers if t == 2)
        n_at_t1 = sum(1 for t in tiers if t == 1)
        n_at_t0 = sum(1 for t in tiers if t == 0)
        op_summary.append({
            "op_id": op_id, "max_tier": max_tier, "max_d": max_d,
            "n_at_t2": n_at_t2, "n_at_t1": n_at_t1, "n_at_t0": n_at_t0,
            "any_escalated": any_escalated,
        })
    op_summary.sort(key=lambda r: r["max_d"], reverse=True)

    n_escalated = sum(1 for r in op_summary if r["any_escalated"])
    n_total = len(op_summary)
    print(f"  ops escalated at any point: {n_escalated} / {n_total}")
    print(f"  ops never escalated:        {n_total - n_escalated} / {n_total}")
    print()
    print("  TOP 8 by max disagreement:")
    for r in op_summary[:8]:
        print(f"    {r['op_id']:<28} max_tier={r['max_tier']}  "
              f"max_d={r['max_d']:.4f}  "
              f"t0/t1/t2={r['n_at_t0']}/{r['n_at_t1']}/{r['n_at_t2']}")
    print()
    print("  BOTTOM 8 by max disagreement:")
    for r in op_summary[-8:]:
        print(f"    {r['op_id']:<28} max_tier={r['max_tier']}  "
              f"max_d={r['max_d']:.4f}  "
              f"t0/t1/t2={r['n_at_t0']}/{r['n_at_t1']}/{r['n_at_t2']}")
    print()

    # Bank tier distribution at end.
    bank_state = bank.active_state()
    tier_counts: dict[int, int] = defaultdict(int)
    for op_id, t in bank_state.items():
        tier_counts[t] += 1
    print(f"  bank tier distribution at end: {dict(sorted(tier_counts.items()))}")
    print()

    # Logits comparison: if generation diverged, last-step logits won't
    # be on the same vocab position. We compare them only if both phases
    # ended at the "same step" (which is just step max_new_tokens-1).
    # The argmax may differ — that's expected.
    print("=" * 70)
    print("Phase A vs Phase B comparison")
    print("=" * 70)
    if result_a["tokens"] == result_b["tokens"]:
        print(f"  IDENTICAL token sequences.")
    else:
        # Find first divergence.
        diverge = next(
            (i for i, (a, b) in enumerate(zip(result_a["tokens"], result_b["tokens"]))
             if a != b),
            min(len(result_a["tokens"]), len(result_b["tokens"])),
        )
        print(f"  Token sequences DIFFER. First divergence at step {diverge}.")
        print(f"    A: {result_a['tokens'][:diverge+3]}")
        print(f"    B: {result_b['tokens'][:diverge+3]}")
    cos = cosine_similarity(final_logits_a, final_logits_b)
    diff = max_abs_diff(final_logits_a, final_logits_b)
    print(f"  Final-logits cosine similarity: {cos:.6f}")
    print(f"  Final-logits max abs diff:      {diff:.4e}")
    print()

    # ------------------------------------------------------------------
    # Hard assertions.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Assertions")
    print("=" * 70)

    failures = []
    nonzero_obs = [d for vals in disagreements.values() for d in vals if d > 0]

    # (1) Both phases generated max_new_tokens.
    print(f"  (1) Phase A generated {len(result_a['tokens'])}/{max_new_tokens}, "
          f"Phase B generated {len(result_b['tokens'])}/{max_new_tokens}")
    if (len(result_a["tokens"]) != max_new_tokens
            or len(result_b["tokens"]) != max_new_tokens):
        failures.append("generation didn't complete; cache/state corruption?")

    # (2) Verifier produces non-constant signal after warmup.
    if not nonzero_obs:
        failures.append("verifier never produced a nonzero value (warmup never ended?)")
    else:
        unique_vals = len(set(round(d, 3) for d in nonzero_obs))
        print(f"  (2) post-warmup unique disagreement values (rounded to .001): "
              f"{unique_vals}")
        if unique_vals < 5:
            failures.append(
                f"verifier produced only {unique_vals} distinct values — "
                "likely saturated or constant"
            )

    # (3) Some ops escalated AND some did not.
    print(f"  (3) escalated: {n_escalated}/{n_total} ops escalated at any point")
    if n_escalated == 0:
        failures.append("no ops escalated at all — verifier didn't drive anything")
    elif n_escalated == n_total:
        failures.append(
            f"ALL {n_total} ops escalated — verifier is a panic button, "
            "not a discriminator"
        )

    # (4) Final bank state has a mix.
    n_above_zero = sum(1 for t in bank_state.values() if t > 0)
    n_at_zero = sum(1 for t in bank_state.values() if t == 0)
    print(f"  (4) bank state mix: {n_above_zero} ops > tier 0, {n_at_zero} ops at tier 0")
    if n_above_zero == 0:
        failures.append("bank never moved any op")
    if n_at_zero == 0 and n_total > 0:
        failures.append("bank moved EVERY op — same panic-button failure")

    # (5) Phase A vs B differs.
    print(f"  (5) Phase A vs B token divergence or final-logit divergence")
    if result_a["tokens"] == result_b["tokens"] and diff < 0.01:
        failures.append(
            "Phase A and Phase B produced identical results — escalations "
            "had no effect on the actual computation"
        )

    print()
    if failures:
        print("✗ FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("✓ PASSED")
    print(f"  Verifier produced ranked, non-saturated signal across {len(nonzero_obs)} "
          f"observations.")
    print(f"  Controller escalated {n_escalated}/{n_total} ops; bank reflects "
          f"{n_above_zero} ops at tier > 0.")
    print(f"  Adaptive generation produced different behavior than fixed-tier-0 "
          f"baseline.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default="mlx-community/Qwen2.5-1.5B-Instruct-bf16",
    )
    p.add_argument("--probes", required=True)
    p.add_argument(
        "--prompt",
        default="Explain why SSD streaming is the bottleneck for local LLM inference.",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--no-runtime-stats", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    return run_test(
        model_id=args.model,
        probes_path=args.probes,
        prompt=args.prompt,
        threshold=args.threshold,
        warmup=args.warmup,
        max_new_tokens=args.max_new_tokens,
        runtime_stats=not args.no_runtime_stats,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
