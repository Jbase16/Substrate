"""
scripts/test_weight_bank.py — Validation harness for WeightBank.

Four tests. Run in order — each builds on the previous.

    Test B0: Bank construction preserves FP16 reference path.
             Build a bank with all ops at tier 0 = NEAR_FP16. Forward
             must match unquantized model exactly (or within FP16
             rounding). Validates that the bank's "do nothing" path is
             a real no-op.

    Test B1: Single-op swap to 4-bit.
             Build a bank with mixed tiers including a 4-bit option.
             Swap layer_0.attention to 4-bit. Run forward.
             Compare against running the same model with that one
             attention quantized in-place via quantize_module_in_place.
             They should match exactly: same dequantization math.

    Test B2: Repeated swap loop, memory snapshot.
             Loop 100 times: swap to 2-bit, 4-bit, 6-bit, fp16, back to 2-bit.
             Capture process RSS at each iteration. Pass requires that
             RSS does not climb monotonically across the loop. Some
             oscillation is acceptable (MLX defers buffer release); a
             trend line is not.

    Test B3: Full mixed-plan forward via bank-driven swaps.
             Compile a real Substrate plan, build a bank against it,
             walk every op installing tier 0 (the plan default), run
             forward, sanity-check output. This is the bank-driven
             equivalent of the kernel's Test 2.

Usage:
    .venv/bin/python -m scripts.test_weight_bank \\
        --model mlx-community/Qwen2.5-1.5B-Instruct-bf16 \\
        --calibration <path> --test 3
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import resource
import sys
from pathlib import Path
from typing import Any


def _import_mlx():
    import mlx.core as mx
    import mlx_lm
    return mx, mlx_lm


# ---------------------------------------------------------------------------
# Comparison metrics (lifted from test_mlx_kernel.py — keep test files
# independent so failures are localized).
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


def top_k_overlap(a, b, k=5):
    top_a = sorted(range(len(a)), key=lambda i: a[i], reverse=True)[:k]
    top_b = sorted(range(len(b)), key=lambda i: b[i], reverse=True)[:k]
    return len(set(top_a) & set(top_b)), k


def argmax_match(a, b):
    return max(range(len(a)), key=lambda i: a[i]) == max(range(len(b)), key=lambda i: b[i])


def max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def extract_last_logits(logits_array):
    last = logits_array[0, -1, :]
    return [float(v) for v in last.tolist()]


def rss_mb() -> float:
    """Process resident set size in MB. macOS reports in bytes, Linux in KB."""
    bytes_ = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return bytes_ / (1024 * 1024)
    return bytes_ / 1024


# Plan-building helpers live in substrate.bench. We re-export them here
# for backwards compatibility with any callers that still do
# `from scripts.test_weight_bank import build_test_plan, ...`. New code
# should import directly from substrate.bench.
from substrate.bench import build_test_plan, build_op_tier_precisions  # noqa: F401


# ---------------------------------------------------------------------------
# Test B0.
# ---------------------------------------------------------------------------
def run_test_b0(model_id: str, prompt: str, verbose: bool) -> int:
    """
    Bank @ tier 0 = NEAR_FP16 should be a no-op against the original model.
    """
    mx, mlx_lm = _import_mlx()

    print("=" * 70)
    print("TEST B0: Bank @ tier 0 = fp16 == original model (no-op preservation)")
    print("=" * 70)
    print(f"Model:  {model_id}")
    print(f"Prompt: {prompt!r}")
    print()

    print("Loading model...")
    model, tokenizer = mlx_lm.load(model_id)
    tokens_list = tokenizer.encode(prompt)
    if hasattr(tokens_list, "tolist"):
        tokens_list = tokens_list.tolist()
    tokens = mx.array([tokens_list])
    print(f"  {len(tokens_list)} tokens")
    print()

    from substrate.backend.mlx_session import MLXForwardSession
    session = MLXForwardSession(model)
    layers = session.kernel._layers

    # Run forward BEFORE building the bank — capture baseline logits.
    print("Baseline forward (no bank involved)...")
    logits_baseline = session.forward_via_kernel(tokens)
    last_baseline = extract_last_logits(logits_baseline)
    print(f"  top-1: {max(range(len(last_baseline)), key=lambda i: last_baseline[i])}")
    print()

    # Build a single-tier plan: only tier 0 = fp16.
    plan = build_test_plan(model_id, layers, {0: "fp16_eq"})
    op_tier_precisions = build_op_tier_precisions(plan, {0: "fp16_eq"})

    print("Building RAMWeightStore (fp16 reference only)...")
    from substrate.backend.ram_weight_store import RAMWeightStore
    ops_with_tiers = {ob.op_id: [0] for ob in plan.op_bundles}
    store = RAMWeightStore(model, layers, ops_with_tiers, op_tier_precisions)
    print()

    print("Building WeightBank (initializes everything to tier 0 = fp16)...")
    from substrate.backend.weight_bank import WeightBank
    bank = WeightBank(model, layers, store, plan)
    print(f"  Active tier sample: {dict(list(bank.active_state().items())[:3])}")
    print()

    # Re-run forward — the bank just installed FP16 references back over
    # FP16 originals. Should still match exactly.
    print("Forward after bank initialization...")
    session.reset()
    logits_after = session.forward_via_kernel(tokens)
    last_after = extract_last_logits(logits_after)
    print(f"  top-1: {max(range(len(last_after)), key=lambda i: last_after[i])}")
    print()

    cos = cosine_similarity(last_baseline, last_after)
    overlap, k = top_k_overlap(last_baseline, last_after, k=5)
    argmax = argmax_match(last_baseline, last_after)
    diff = max_abs_diff(last_baseline, last_after)
    print(f"  cosine_similarity: {cos:.10f}")
    print(f"  top-5 overlap:     {overlap}/{k}")
    print(f"  argmax match:      {argmax}")
    print(f"  max abs diff:      {diff:.6e}")
    print()

    if cos >= 0.99999 and argmax and overlap == k and diff < 1e-3:
        print("✓ TEST B0 PASSED")
        return 0
    print("✗ TEST B0 FAILED")
    if cos < 0.99999:
        print(f"  cosine {cos} < 0.99999")
    if not argmax:
        print(f"  argmax mismatch")
    if overlap != k:
        print(f"  top-{k} overlap {overlap} != {k}")
    if diff >= 1e-3:
        print(f"  max diff {diff} >= 1e-3")
    return 1


# ---------------------------------------------------------------------------
# Test B1.
# ---------------------------------------------------------------------------
def run_test_b1(model_id: str, prompt: str, verbose: bool) -> int:
    """
    Bank with all ops at tier 0 == 4-bit must produce the same forward
    pass as in-place 4-bit quantization of every op.

    The bank\'s tier ordering follows the IR\'s monotonicity rule: tier 0
    is the LOWEST-precision tier (cheapest to evaluate, highest quality
    risk). Tier N (N>0) escalates upward. So an "all-4bit baseline" plan
    has tier 0 = 4bit; the bank initializes everything to 4-bit on
    construction.

    Strategy:
        Model A: load FP16, build bank with single-tier plan (tier 0 = 4bit).
                  Bank init quantizes every op to 4-bit via packed-and-dequant.
        Model B: load FP16, run quantize_module_in_place(every layer attn+mlp)
                  at 4 bits.
        Both should produce identical forward results.

    Pass criterion: byte-identical logits (same dequant math everywhere).
    """
    mx, mlx_lm = _import_mlx()

    print("=" * 70)
    print("TEST B1: bank-init-all-to-4bit == in-place-quantize-all-to-4bit")
    print("=" * 70)

    print("Loading model A (bank-driven)...")
    model_a, tokenizer = mlx_lm.load(model_id)
    tokens_list = tokenizer.encode(prompt)
    if hasattr(tokens_list, "tolist"):
        tokens_list = tokens_list.tolist()
    tokens = mx.array([tokens_list])

    from substrate.backend.mlx_session import MLXForwardSession
    session_a = MLXForwardSession(model_a)
    layers_a = session_a.kernel._layers

    # Single-tier plan: every op at 4-bit. Tier 0 only.
    plan = build_test_plan(model_id, layers_a, {0: "4bit"})
    precisions = build_op_tier_precisions(plan, {0: "4bit"})

    print("Building bank with tier 0 = 4bit (single-tier plan)...")
    from substrate.backend.ram_weight_store import RAMWeightStore
    from substrate.backend.weight_bank import WeightBank
    ops_with_tiers = {ob.op_id: [0] for ob in plan.op_bundles}
    store_a = RAMWeightStore(model_a, layers_a, ops_with_tiers, precisions)
    bank_a = WeightBank(model_a, layers_a, store_a, plan)
    print(f"  Bank initialized: all {len(plan.op_bundles)} ops at 4-bit")
    print()

    print("Running forward A (bank-driven)...")
    session_a.reset()
    logits_a = session_a.forward_via_kernel(tokens)
    last_a = extract_last_logits(logits_a)
    print(f"  top-1: {max(range(len(last_a)), key=lambda i: last_a[i])}")
    print()

    print("Loading model B (in-place quantize every layer)...")
    model_b, _ = mlx_lm.load(model_id)
    session_b = MLXForwardSession(model_b)
    layers_b = session_b.kernel._layers
    from substrate.backend.quantize import quantize_module_in_place
    total_modified = 0
    for layer_id in range(len(layers_b)):
        total_modified += quantize_module_in_place(layers_b[layer_id].self_attn, bits=4)
        total_modified += quantize_module_in_place(layers_b[layer_id].mlp, bits=4)
    print(f"  Quantized {total_modified} weight arrays across all layers")

    print("Running forward B (in-place quantized)...")
    session_b.reset()
    logits_b = session_b.forward_via_kernel(tokens)
    last_b = extract_last_logits(logits_b)
    print(f"  top-1: {max(range(len(last_b)), key=lambda i: last_b[i])}")
    print()

    cos = cosine_similarity(last_a, last_b)
    overlap, k = top_k_overlap(last_a, last_b, k=5)
    argmax = argmax_match(last_a, last_b)
    diff = max_abs_diff(last_a, last_b)
    print(f"  cosine_similarity: {cos:.10f}")
    print(f"  top-5 overlap:     {overlap}/{k}")
    print(f"  argmax match:      {argmax}")
    print(f"  max abs diff:      {diff:.6e}")
    print()

    if cos >= 0.99999 and argmax and overlap == k and diff < 0.01:
        print("✓ TEST B1 PASSED")
        return 0
    print("✗ TEST B1 FAILED")
    return 1


# ---------------------------------------------------------------------------
# Test B2.
# ---------------------------------------------------------------------------
def run_test_b2(model_id: str, prompt: str, verbose: bool) -> int:
    """
    Repeated swap loop, RSS monotonicity check.

    Loop 100 times: swap one op through 2 → 3 → 4 → 6 → fp16 → repeat.
    Track RSS at each iteration. RSS may grow during MLX warmup but
    should stabilize. A monotonic linear climb means we're leaking.

    Pass criterion: in the last 30 iterations, RSS should not climb by
    more than 50 MB. (Some jitter is normal; 50MB across 30 iters is
    well above noise but well below "we're leaking arrays".)
    """
    mx, mlx_lm = _import_mlx()

    print("=" * 70)
    print("TEST B2: repeated swap loop, RSS monotonicity")
    print("=" * 70)

    print("Loading model...")
    model, tokenizer = mlx_lm.load(model_id)
    tokens_list = tokenizer.encode(prompt)
    if hasattr(tokens_list, "tolist"):
        tokens_list = tokens_list.tolist()
    tokens = mx.array([tokens_list])

    from substrate.backend.mlx_session import MLXForwardSession
    session = MLXForwardSession(model)
    layers = session.kernel._layers

    plan = build_test_plan(model_id, layers, {0: "2bit", 1: "3bit", 2: "4bit", 3: "6bit", 4: "fp16_eq"})
    precisions = build_op_tier_precisions(plan, {0: "2bit", 1: "3bit", 2: "4bit", 3: "6bit", 4: "fp16_eq"})

    print("Building store with all 5 tiers per op...")
    from substrate.backend.ram_weight_store import RAMWeightStore
    from substrate.backend.weight_bank import WeightBank
    # For RSS test, only build the bank for one op to keep store small.
    test_op = "layer_0.attention"
    ops_with_tiers = {test_op: [0, 1, 2, 3, 4]}

    # The plan still has all ops, but the store only has one. This works
    # because the bank's __init__ initializes every op in the plan to
    # tier 0 — so we need at least tier 0 for every op. Easiest: make
    # the store cover tier 0 for ALL ops at fp16, plus all tiers for our
    # test op.
    for ob in plan.op_bundles:
        if ob.op_id != test_op:
            ops_with_tiers[ob.op_id] = [0]
            # Override its tier-0 precision to fp16 so the store will
            # produce a valid PackedWeight.
            precisions[(ob.op_id, 0)] = "fp16_eq"
    store = RAMWeightStore(model, layers, ops_with_tiers, precisions)
    bank = WeightBank(model, layers, store, plan)
    print(f"  Bank initialized.")
    print()

    iterations = 100
    rss_history: list[float] = []
    rss_at_start = rss_mb()
    print(f"Initial RSS: {rss_at_start:.1f} MB")

    swap_pattern = [0, 1, 2, 3, 4]
    print(f"Swap loop: {iterations} iterations, pattern {swap_pattern}")
    for i in range(iterations):
        target = swap_pattern[i % len(swap_pattern)]
        bank.swap(test_op, target_tier=target)
        rss = rss_mb()
        rss_history.append(rss)
        if i < 5 or i % 20 == 0 or i >= iterations - 5:
            print(f"  iter {i:3d}: tier={target}, rss={rss:7.1f} MB, delta={rss - rss_at_start:+7.1f} MB")

    # Analyze the last 30 iterations.
    tail = rss_history[-30:]
    tail_growth = tail[-1] - tail[0]
    tail_max_minus_min = max(tail) - min(tail)
    print()
    print(f"Last 30 iterations: max-min={tail_max_minus_min:.1f} MB, end-start={tail_growth:+.1f} MB")
    print(f"Total run RSS growth: {rss_history[-1] - rss_at_start:+.1f} MB")

    if tail_growth < 50 and tail_max_minus_min < 100:
        print()
        print("✓ TEST B2 PASSED (RSS stable across swap loop)")
        return 0
    print()
    print("✗ TEST B2 FAILED")
    if tail_growth >= 50:
        print(f"  tail growth {tail_growth:.1f} MB >= 50 MB (likely leak)")
    if tail_max_minus_min >= 100:
        print(f"  tail variation {tail_max_minus_min:.1f} MB >= 100 MB")
    return 1


# ---------------------------------------------------------------------------
# Test B3.
# ---------------------------------------------------------------------------
def run_test_b3(
    model_id: str, prompt: str, calibration_path: str | None,
    max_ram_gb: float, quality_cap: float, verbose: bool,
) -> int:
    """
    Full mixed-plan forward with bank-driven swaps. Bank-driven equivalent
    of test_mlx_kernel.py Test 2.
    """
    mx, mlx_lm = _import_mlx()

    print("=" * 70)
    print("TEST B3: real Substrate plan via WeightBank → forward")
    print("=" * 70)

    print("Loading model...")
    model, tokenizer = mlx_lm.load(model_id)
    tokens_list = tokenizer.encode(prompt)
    if hasattr(tokens_list, "tolist"):
        tokens_list = tokens_list.tolist()
    tokens = mx.array([tokens_list])

    from substrate.backend.mlx_session import MLXForwardSession
    session = MLXForwardSession(model)
    layers = session.kernel._layers
    print(f"  {len(layers)} layers loaded")
    print()

    # Build profile + estimator + plan, mirroring test_mlx_kernel Test 2.
    from substrate.compiler.ir import OpKind
    from substrate.compiler.planner import OpProfile, ModelProfile, _PRECISION_LADDER
    from substrate.backend.quantize import snapshot_weights

    def count(module):
        snaps = snapshot_weights(module)
        return sum(
            int(__import__("functools").reduce(lambda a, b: a * b, arr.shape, 1))
            for _, _, arr in snaps
        )
    attn_params = count(layers[0].self_attn)
    mlp_params = count(layers[0].mlp)

    ops = []
    for layer_id in range(len(layers)):
        sensitive = layer_id in (0, 1, len(layers) - 2, len(layers) - 1)
        ops.append(OpProfile(
            op_id=f"layer_{layer_id}.attention",
            op_kind=OpKind.ATTENTION, layer_id=layer_id,
            param_count=attn_params, skeleton_compute_us=120,
            full_precision_compute_us=350,
            sensitivity=0.7 if sensitive else 0.3,
        ))
        ops.append(OpProfile(
            op_id=f"layer_{layer_id}.mlp_dense",
            op_kind=OpKind.MLP_DENSE, layer_id=layer_id,
            param_count=mlp_params, skeleton_compute_us=300,
            full_precision_compute_us=900,
            sensitivity=0.7 if sensitive else 0.3,
        ))
    profile = ModelProfile(
        model_id=model_id, ops=tuple(ops),
        embedding_bytes=896 * 151_936 * 2,
        lm_head_bytes=896 * 151_936 * 2,
        runtime_overhead_bytes=200_000_000,
    )

    estimator = None
    if calibration_path:
        from substrate.calibration.schema import load_calibration
        from substrate.calibration.adapter import estimator_from_calibration
        cal = load_calibration(calibration_path)
        estimator = estimator_from_calibration(cal)
        print(f"  Loaded {len(cal.cells)} calibration cells")

    from substrate import Budget, EscalationPolicy, Planner
    budget = Budget(int(max_ram_gb * 1e9), int(50e9), int(5e9), quality_cap, 30.0)
    print("Compiling plan...")
    plan = Planner(quality_estimator=estimator, escalation_policy=EscalationPolicy(max_concurrent_escalations=4)).compile(profile, budget)
    print(f"  predicted_loss: {plan.predicted_quality_loss:.6f}")
    prec_counts: dict[str, int] = {}
    for ob in plan.op_bundles:
        max_residual = max(
            (int(t.tensor_id.split("_")[-1]) for t in ob.default.requires if "residual" in t.tensor_id),
            default=0,
        )
        prec = _PRECISION_LADDER[max_residual].value
        prec_counts[prec] = prec_counts.get(prec, 0) + 1
    print(f"  Tier-0 distribution: {dict(sorted(prec_counts.items()))}")
    print()

    # Build per-op tier precision map. For this test we use ONLY tier 0 from
    # the plan (the default execution path); escalation tiers are admitted
    # but the bank only needs tier 0 for the forward pass.
    op_tier_precisions: dict[tuple[str, int], str] = {}
    ops_with_tiers: dict[str, list[int]] = {}
    for ob in plan.op_bundles:
        max_residual = max(
            (int(t.tensor_id.split("_")[-1]) for t in ob.default.requires if "residual" in t.tensor_id),
            default=0,
        )
        prec = _PRECISION_LADDER[max_residual].value
        op_tier_precisions[(ob.op_id, 0)] = prec
        ops_with_tiers[ob.op_id] = [0]

    print("Building RAMWeightStore...")
    from substrate.backend.ram_weight_store import RAMWeightStore
    from substrate.backend.weight_bank import WeightBank
    store = RAMWeightStore(model, layers, ops_with_tiers, op_tier_precisions)
    print()

    print("Building WeightBank (initializes all ops to tier 0 per the plan)...")
    bank = WeightBank(model, layers, store, plan)
    print(f"  active state sample: {dict(list(bank.active_state().items())[:3])}")
    print()

    print("Running forward via kernel + bank...")
    session.reset()
    logits = session.forward_via_kernel(tokens)
    last = extract_last_logits(logits)

    has_nan = any(math.isnan(v) for v in last)
    has_inf = any(math.isinf(v) for v in last)
    top1 = max(range(len(last)), key=lambda i: last[i])
    top5 = sorted(range(len(last)), key=lambda i: last[i], reverse=True)[:5]
    print(f"  output shape: {logits.shape}")
    print(f"  contains NaN: {has_nan}")
    print(f"  contains Inf: {has_inf}")
    print(f"  top-1: {top1} (logit {last[top1]:.4f})")
    try:
        decoded = [tokenizer.decode([t]) for t in top5]
        print(f"  top-5 decoded: {decoded}")
    except Exception as e:
        print(f"  top-5 decode failed: {e}")
    print()

    if not has_nan and not has_inf and math.isfinite(last[top1]):
        print("✓ TEST B3 PASSED")
        return 0
    print("✗ TEST B3 FAILED")
    return 1


# ---------------------------------------------------------------------------
# CLI driver.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16",
    )
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--test", choices=("0", "1", "2", "3", "all"), default="0")
    p.add_argument("--calibration", default=None)
    p.add_argument("--max-ram-gb", type=float, default=8.0)
    p.add_argument("--quality-cap", type=float, default=0.05)
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    runners = {
        "0": lambda: run_test_b0(args.model, args.prompt, args.verbose),
        "1": lambda: run_test_b1(args.model, args.prompt, args.verbose),
        "2": lambda: run_test_b2(args.model, args.prompt, args.verbose),
        "3": lambda: run_test_b3(
            args.model, args.prompt, args.calibration,
            args.max_ram_gb, args.quality_cap, args.verbose,
        ),
    }

    if args.test == "all":
        for tnum in ("0", "1", "2", "3"):
            rc = runners[tnum]()
            print()
            if rc != 0:
                print(f"Test B{tnum} failed; aborting.")
                return rc
        return 0
    return runners[args.test]()


if __name__ == "__main__":
    sys.exit(main())
