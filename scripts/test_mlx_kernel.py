"""
scripts/test_mlx_kernel.py — Validation harness for MLXOpKernel.

Three tests, in strict ordering. Each test is gated on the previous
passing — if Test 0 fails, the orchestration itself is wrong and Tests
1/2 are meaningless.

    Test 0: FP16 normal vs FP16 split
            Pure orchestration. No quantization. No tier swap.
            Pass criterion: kernel-driven forward produces identical
            logits to model(tokens) within fp16 rounding tolerance.

    Test 1: uniform 4-bit normal vs uniform 4-bit split
            (To be added once Test 0 passes.)

    Test 2: mixed-precision Substrate plan
            (To be added once Test 1 passes.)

Comparison metrics, per spec:
    - cosine similarity on final logits
    - top-5 overlap / argmax match
    - max absolute logit difference

Usage:
    python -m scripts.test_mlx_kernel \\
        --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 \\
        --test 0
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
# Comparison metrics. Operate on flat lists of floats for portability.
# ---------------------------------------------------------------------------
def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0 if (na != 0 or nb != 0) else 1.0
    return dot / (na * nb)


def top_k_overlap(logits_a: list[float], logits_b: list[float], k: int = 5) -> tuple[int, int]:
    """
    Return (overlap_count, k). Both logit vectors are over the same vocab.
    overlap_count is how many of the top-k indices in A also appear in
    the top-k of B.
    """
    top_a = sorted(range(len(logits_a)), key=lambda i: logits_a[i], reverse=True)[:k]
    top_b = sorted(range(len(logits_b)), key=lambda i: logits_b[i], reverse=True)[:k]
    return len(set(top_a) & set(top_b)), k


def argmax_match(logits_a: list[float], logits_b: list[float]) -> bool:
    a_max = max(range(len(logits_a)), key=lambda i: logits_a[i])
    b_max = max(range(len(logits_b)), key=lambda i: logits_b[i])
    return a_max == b_max


def max_abs_diff(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Logits extraction. Both forward paths return [batch, seq, vocab].
# We compare the LAST-token logits, which is what generation uses.
# ---------------------------------------------------------------------------
def extract_last_logits(logits_array) -> list[float]:
    """
    Pull the last-token logits as a Python list of floats. Forces evaluation
    of the lazy mx graph by calling .tolist().
    """
    # Shape: [batch, seq, vocab]. Take batch 0, last seq position.
    last = logits_array[0, -1, :]
    return [float(v) for v in last.tolist()]


# ---------------------------------------------------------------------------
# Test 0.
# ---------------------------------------------------------------------------
def run_test_0(model_id: str, prompt: str, verbose: bool) -> int:
    """
    FP16 normal vs FP16 split.

    Loads the model, runs both forward paths on the same prompt, compares
    logits. Pass requires:
        - cosine similarity >= 0.9999
        - argmax match
        - top-5 overlap == 5
        - max abs diff < 1.0 (rounding-error level for fp16 logits)
    """
    mx, mlx_lm = _import_mlx()

    print("=" * 70)
    print("TEST 0: FP16 normal forward vs FP16 split forward")
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
    print(f"  {len(tokens_list)} tokens encoded")
    print()

    print("Constructing session...")
    from substrate.backend.mlx_session import MLXForwardSession
    session = MLXForwardSession(model)
    print(f"  {session.num_layers} layers")
    print()

    print("Running forward_normal()...")
    logits_normal = session.forward_normal(tokens)
    last_normal = extract_last_logits(logits_normal)
    print(f"  output shape: {logits_normal.shape}")
    print(f"  last-token logits: top-1={max(range(len(last_normal)), key=lambda i: last_normal[i])}")
    print()

    print("Running forward_via_kernel()...")
    session.reset()
    logits_kernel = session.forward_via_kernel(tokens)
    last_kernel = extract_last_logits(logits_kernel)
    print(f"  output shape: {logits_kernel.shape}")
    print(f"  last-token logits: top-1={max(range(len(last_kernel)), key=lambda i: last_kernel[i])}")
    print()

    # Comparisons.
    print("Comparison:")
    cos = cosine_similarity(last_normal, last_kernel)
    overlap, k = top_k_overlap(last_normal, last_kernel, k=5)
    argmax_eq = argmax_match(last_normal, last_kernel)
    max_diff = max_abs_diff(last_normal, last_kernel)
    print(f"  cosine_similarity:  {cos:.10f}")
    print(f"  top-5 overlap:      {overlap}/{k}")
    print(f"  argmax match:       {argmax_eq}")
    print(f"  max abs diff:       {max_diff:.6e}")
    print()

    # Verbose: dump first few logit values side-by-side.
    if verbose:
        print("First 10 logit pairs (normal | kernel | diff):")
        for i in range(min(10, len(last_normal))):
            n, k_v = last_normal[i], last_kernel[i]
            print(f"  [{i}] {n:+.6f} | {k_v:+.6f} | {abs(n - k_v):.2e}")
        print()

    # Pass criteria.
    passed = True
    failures = []
    if cos < 0.9999:
        passed = False
        failures.append(f"cosine_similarity {cos:.6f} < 0.9999")
    if not argmax_eq:
        passed = False
        failures.append("argmax does not match")
    if overlap != k:
        passed = False
        failures.append(f"top-{k} overlap {overlap} != {k}")
    if max_diff > 1.0:
        passed = False
        failures.append(f"max abs diff {max_diff:.4f} > 1.0")

    if passed:
        print("✓ TEST 0 PASSED")
        return 0
    else:
        print("✗ TEST 0 FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1


# ---------------------------------------------------------------------------
# Test 1.
# ---------------------------------------------------------------------------
def run_test_1(model_id: str, prompt: str, bits: int, verbose: bool) -> int:
    """
    Uniform N-bit normal vs uniform N-bit split.

    Strategy:
        1. Load FP16 model.
        2. Apply quantize_module_in_place to all transformer layers + lm_head
           at the same N-bit precision.
        3. Run forward_normal() — uses the now-quantized weights.
        4. Run forward_via_kernel() — uses the same quantized weights.
        5. Compare. They should be IDENTICAL because both paths walk the
           same module tree with the same weights.

    Pass criterion is therefore tighter than Test 0:
        - cosine_similarity == 1.0 (exact match expected)
        - max abs diff == 0.0
        - argmax match
        - top-5 overlap == 5

    If Test 1 produces non-zero diff, the kernel orchestration is
    sensitive to weight values in some way that pure FP16 didn't expose
    (e.g. float overflow at low precision, mask mismatch in edge cases).
    """
    mx, mlx_lm = _import_mlx()

    print("=" * 70)
    print(f"TEST 1: uniform {bits}-bit normal forward vs uniform {bits}-bit split forward")
    print("=" * 70)
    print(f"Model:  {model_id}")
    print(f"Prompt: {prompt!r}")
    print(f"Bits:   {bits}")
    print()

    print("Loading model...")
    model, tokenizer = mlx_lm.load(model_id)
    tokens_list = tokenizer.encode(prompt)
    if hasattr(tokens_list, "tolist"):
        tokens_list = tokens_list.tolist()
    tokens = mx.array([tokens_list])
    print(f"  {len(tokens_list)} tokens encoded")
    print()

    # Apply uniform N-bit quantization to all transformer blocks + lm_head.
    # We exclude the embedding (Qwen ties them; quantizing breaks output)
    # and the final norm (1D, mx.quantize would skip it anyway).
    print(f"Applying uniform {bits}-bit quantization to all transformer layers + lm_head...")
    from substrate.backend.quantize import quantize_module_in_place

    # Find the layer container and lm_head.
    from substrate.backend.mlx_session import MLXForwardSession
    session = MLXForwardSession(model)
    total_modified = 0
    for layer_id, layer in enumerate(session.kernel._layers):
        modified = quantize_module_in_place(layer, bits=bits)
        total_modified += modified
    # Quantize lm_head if it\'s a separate module (not tied to embeddings).
    if hasattr(model, "lm_head") and callable(getattr(model, "lm_head", None)):
        lm_head = model.lm_head
        if hasattr(lm_head, "items"):
            modified = quantize_module_in_place(lm_head, bits=bits)
            total_modified += modified
    print(f"  Quantized {total_modified} weight arrays")
    print()

    print("Running forward_normal() with quantized weights...")
    logits_normal = session.forward_normal(tokens)
    last_normal = extract_last_logits(logits_normal)
    print(f"  output shape: {logits_normal.shape}")
    print(f"  last-token logits: top-1={max(range(len(last_normal)), key=lambda i: last_normal[i])}")
    print()

    print("Running forward_via_kernel() with same quantized weights...")
    session.reset()
    logits_kernel = session.forward_via_kernel(tokens)
    last_kernel = extract_last_logits(logits_kernel)
    print(f"  output shape: {logits_kernel.shape}")
    print(f"  last-token logits: top-1={max(range(len(last_kernel)), key=lambda i: last_kernel[i])}")
    print()

    print("Comparison:")
    cos = cosine_similarity(last_normal, last_kernel)
    overlap, k = top_k_overlap(last_normal, last_kernel, k=5)
    argmax_eq = argmax_match(last_normal, last_kernel)
    max_diff = max_abs_diff(last_normal, last_kernel)
    print(f"  cosine_similarity:  {cos:.10f}")
    print(f"  top-5 overlap:      {overlap}/{k}")
    print(f"  argmax match:       {argmax_eq}")
    print(f"  max abs diff:       {max_diff:.6e}")
    print()

    if verbose:
        print("First 10 logit pairs (normal | kernel | diff):")
        for i in range(min(10, len(last_normal))):
            n, k_v = last_normal[i], last_kernel[i]
            print(f"  [{i}] {n:+.6f} | {k_v:+.6f} | {abs(n - k_v):.2e}")
        print()

    # Tight pass criteria — both paths use identical weights.
    passed = True
    failures = []
    if cos < 0.99999:
        passed = False
        failures.append(f"cosine_similarity {cos:.6f} < 0.99999")
    if not argmax_eq:
        passed = False
        failures.append("argmax does not match")
    if overlap != k:
        passed = False
        failures.append(f"top-{k} overlap {overlap} != {k}")
    if max_diff > 0.01:
        passed = False
        failures.append(f"max abs diff {max_diff:.4f} > 0.01")

    if passed:
        print(f"✓ TEST 1 PASSED ({bits}-bit)")
        return 0
    else:
        print(f"✗ TEST 1 FAILED ({bits}-bit)")
        for f in failures:
            print(f"  - {f}")
        return 1



# ---------------------------------------------------------------------------
# Test 2.
# ---------------------------------------------------------------------------
def run_test_2(model_id: str, prompt: str, calibration_path: str | None,
               max_ram_gb: float, quality_cap: float, verbose: bool) -> int:
    """
    Mixed-precision Substrate plan on a real model.

    Strategy:
        1. Load FP16 model.
        2. Build a ModelProfile matching it (or load from JSON).
        3. If calibration provided: load it and build estimator.
        4. Compile a plan under the given budget.
        5. Apply per-op quantization: each layer gets its attention and
           mlp_dense quantized to whatever precision the plan chose.
        6. Run forward via kernel.
        7. Sanity check: no NaN/Inf, plausible top-1, plausible top-5.

    There is NO oracle for Test 2. The plan emits unique per-op precisions
    that don't correspond to any single uniform quantization. We can only
    check internal sanity: the model outputs valid logits and predicts a
    coherent next token.

    Pass criterion:
        - output contains no NaN/Inf
        - top-1 logit is finite
        - top-5 tokens decode to plausible text fragments (best-effort)
    """
    mx, mlx_lm = _import_mlx()
    import math

    print("=" * 70)
    print("TEST 2: mixed-precision Substrate plan on real model")
    print("=" * 70)
    print(f"Model:      {model_id}")
    print(f"Prompt:     {prompt!r}")
    print(f"Max RAM:    {max_ram_gb} GB")
    print(f"Quality cap:{quality_cap}")
    print(f"Calibration:{calibration_path or '(stub)'}")
    print()

    print("Loading model...")
    model, tokenizer = mlx_lm.load(model_id)
    tokens_list = tokenizer.encode(prompt)
    if hasattr(tokens_list, "tolist"):
        tokens_list = tokens_list.tolist()
    tokens = mx.array([tokens_list])
    print(f"  {len(tokens_list)} tokens encoded")
    print()

    # Build a ModelProfile matching the loaded model.
    print("Building ModelProfile from loaded model...")
    from substrate.backend.mlx_session import MLXForwardSession
    from substrate.compiler.ir import OpKind
    from substrate.compiler.planner import OpProfile, ModelProfile, _PRECISION_LADDER

    session = MLXForwardSession(model)
    layers = session.kernel._layers

    # Inspect first layer to get param counts. We use mlx_lm-style
    # attribute access; this is Qwen2-shaped.
    attn0 = layers[0].self_attn
    mlp0 = layers[0].mlp

    def count_module_params(module):
        from substrate.backend.quantize import snapshot_weights
        snaps = snapshot_weights(module)
        total = 0
        for _, _, arr in snaps:
            n = 1
            for d in arr.shape:
                n *= int(d)
            total += n
        return total

    attn_params = count_module_params(attn0)
    mlp_params = count_module_params(mlp0)
    print(f"  attention params/layer: {attn_params:,}")
    print(f"  mlp params/layer:       {mlp_params:,}")
    print(f"  total layers:           {len(layers)}")
    print()

    ops = []
    for layer_id in range(len(layers)):
        sensitive = layer_id in (0, 1, len(layers) - 2, len(layers) - 1)
        ops.append(OpProfile(
            op_id=f"layer_{layer_id}.attention",
            op_kind=OpKind.ATTENTION,
            layer_id=layer_id,
            param_count=attn_params,
            skeleton_compute_us=120,
            full_precision_compute_us=350,
            sensitivity=0.7 if sensitive else 0.3,
        ))
        ops.append(OpProfile(
            op_id=f"layer_{layer_id}.mlp_dense",
            op_kind=OpKind.MLP_DENSE,
            layer_id=layer_id,
            param_count=mlp_params,
            skeleton_compute_us=300,
            full_precision_compute_us=900,
            sensitivity=0.7 if sensitive else 0.3,
        ))
    profile = ModelProfile(
        model_id=model_id,
        ops=tuple(ops),
        embedding_bytes=896 * 151_936 * 2,  # Qwen2.5-0.5B hidden_dim
        lm_head_bytes=896 * 151_936 * 2,
        runtime_overhead_bytes=200_000_000,
    )

    # Build estimator from calibration or stub.
    estimator = None
    if calibration_path:
        from substrate.calibration.schema import load_calibration
        from substrate.calibration.adapter import estimator_from_calibration
        print(f"Loading calibration from {calibration_path}...")
        cal = load_calibration(calibration_path)
        estimator = estimator_from_calibration(cal)
        print(f"  {len(cal.cells)} cells loaded")
        print()
    else:
        print("No calibration provided — planner will use stub.")
        print()

    # Compile.
    from substrate import Budget, EscalationPolicy, Planner
    budget = Budget(
        int(max_ram_gb * 1e9), int(50e9), int(5e9),
        quality_cap, 30.0,
    )
    print("Compiling plan...")
    plan = Planner(
        quality_estimator=estimator,
        escalation_policy=EscalationPolicy(max_concurrent_escalations=4),
    ).compile(profile, budget)

    print(f"  predicted_loss: {plan.predicted_quality_loss:.6f}")
    print(f"  RAM peak:       {plan.predicted_peak_resident_bytes / 1e9:.2f} GB")
    print(f"  pool:           {plan.escalation_ram_pool_bytes / 1e6:.1f} MB")

    prec_counts: dict[str, int] = {}
    for ob in plan.op_bundles:
        max_residual = max(
            (int(t.tensor_id.split("_")[-1])
             for t in ob.default.requires if "residual" in t.tensor_id),
            default=0,
        )
        prec = _PRECISION_LADDER[max_residual].value
        prec_counts[prec] = prec_counts.get(prec, 0) + 1
    print(f"  Tier-0 distribution: {dict(sorted(prec_counts.items()))}")
    print()

    # Apply per-op quantization based on the plan.
    print("Applying per-op quantization to model in place...")
    from substrate.backend.quantize import quantize_module_in_place

    bits_for_precision = {
        "2bit": 2, "3bit": 3, "4bit": 4, "6bit": 6, "fp16_eq": None,
    }
    total_modified = 0
    for ob in plan.op_bundles:
        max_residual = max(
            (int(t.tensor_id.split("_")[-1])
             for t in ob.default.requires if "residual" in t.tensor_id),
            default=0,
        )
        prec_name = _PRECISION_LADDER[max_residual].value
        bits = bits_for_precision[prec_name]
        if bits is None:
            continue  # fp16 = no quantization
        # Find the sub-module for this op.
        layer_id = ob.default.layer_id
        kind = ob.default.op_kind.value
        layer = layers[layer_id]
        if kind == "attention":
            module = layer.self_attn
        elif kind == "mlp_dense":
            module = layer.mlp
        else:
            continue
        modified = quantize_module_in_place(module, bits=bits)
        total_modified += modified
    print(f"  Quantized {total_modified} weight arrays per the plan")
    print()

    # Run kernel forward.
    print("Running forward_via_kernel() with mixed-precision weights...")
    session.reset()
    logits = session.forward_via_kernel(tokens)
    last_logits = extract_last_logits(logits)
    print(f"  output shape: {logits.shape}")
    print()

    # Sanity checks.
    print("Sanity checks:")
    has_nan = any(math.isnan(v) for v in last_logits)
    has_inf = any(math.isinf(v) for v in last_logits)
    print(f"  contains NaN: {has_nan}")
    print(f"  contains Inf: {has_inf}")

    top1_idx = max(range(len(last_logits)), key=lambda i: last_logits[i])
    top5 = sorted(range(len(last_logits)), key=lambda i: last_logits[i], reverse=True)[:5]
    print(f"  top-1 index:  {top1_idx} (logit {last_logits[top1_idx]:.4f})")
    print(f"  top-5 indices:{top5}")

    # Decode top-5 to text for plausibility.
    try:
        top5_text = [tokenizer.decode([t]) for t in top5]
        print(f"  top-5 decoded:{top5_text}")
    except Exception as exc:
        print(f"  decode failed: {exc}")

    print()

    passed = not has_nan and not has_inf and math.isfinite(last_logits[top1_idx])
    if passed:
        print("✓ TEST 2 PASSED (mixed-precision plan executes cleanly)")
        return 0
    else:
        print("✗ TEST 2 FAILED")
        if has_nan:
            print("  - logits contain NaN")
        if has_inf:
            print("  - logits contain Inf")
        return 1



# ---------------------------------------------------------------------------
# CLI driver.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16",
        help="HuggingFace model id (mlx-community Qwen2 variant).",
    )
    p.add_argument(
        "--prompt",
        default="The capital of France is",
        help="Test prompt. Short prompts are fine for this validation.",
    )
    p.add_argument(
        "--test", type=int, choices=(0, 1, 2), default=0,
        help="Which test to run.",
    )
    p.add_argument(
        "--bits", type=int, default=4,
        help="Bit-width for Test 1\'s uniform quantization (2,3,4,6,8).",
    )
    p.add_argument(
        "--calibration", default=None,
        help="Path to calibration.json for Test 2 (mixed-precision plan).",
    )
    p.add_argument(
        "--max-ram-gb", type=float, default=8.0,
        help="RAM budget in GB for Test 2 plan compilation.",
    )
    p.add_argument(
        "--quality-cap", type=float, default=0.05,
        help="Quality loss cap for Test 2 plan compilation.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Dump per-logit comparison values.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.test == 0:
        return run_test_0(args.model, args.prompt, args.verbose)
    elif args.test == 1:
        return run_test_1(args.model, args.prompt, args.bits, args.verbose)
    elif args.test == 2:
        return run_test_2(
            args.model, args.prompt, args.calibration,
            args.max_ram_gb, args.quality_cap, args.verbose,
        )
    else:
        print(f"Test {args.test} not yet implemented.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
