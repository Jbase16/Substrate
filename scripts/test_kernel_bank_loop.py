"""
scripts/test_kernel_bank_loop.py — Spine test: kernel→bank control loop.

The kernel was wired to a WeightBank such that each call to execute()
asks the bank to install op.tier_index's weights. This test proves that
mechanism works end-to-end:

    Test BC1: same op, two tier_index values, two different forwards.
        Build a 2-tier plan and bank.
        Run forward with all ops at tier 0. Capture logits A.
        Run forward again, but with one op pinned at tier 1 via the
        ScheduledOp's tier_index. Capture logits B.
        Verify:
            (1) A != B (swap actually changed the math)
            (2) B matches an oracle model where exactly that one op was
                quantized to tier-1's precision in place.

If A == B, the kernel/bank wiring is broken: the kernel ignored
tier_index, or the bank failed to install, or both.

If B != oracle, the swap installed but the math diverged from the direct
quantization equivalent — same numerical bug as Test B1 would catch, but
exercised through the runtime control loop instead of the bank's init path.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys


def _import_mlx():
    import mlx.core as mx
    import mlx_lm
    return mx, mlx_lm


# Comparison metrics, shared with other test scripts.
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
# Helper: a tiny ScheduledOp shim. The MLXForwardSession's _SyntheticOp
# only sets tier_index=0, but we need to drive different tier_index per op
# per forward. We replicate the shim and inject custom tier_index values.
# ---------------------------------------------------------------------------
class _OpKindShim:
    __slots__ = ("value",)
    def __init__(self, value): self.value = value


class _RuntimeOp:
    __slots__ = ("op_id", "layer_id", "op_kind", "tier_index")
    def __init__(self, op_id, layer_id, op_kind_value, tier_index):
        self.op_id = op_id
        self.layer_id = layer_id
        self.op_kind = _OpKindShim(op_kind_value)
        self.tier_index = tier_index


def forward_with_tier_overrides(
    session, tokens, tier_overrides: dict[str, int],
):
    """
    Run a forward pass through the kernel where specific ops have their
    tier_index overridden. tier_overrides maps op_id to tier_index; ops
    not in the dict use tier_index=0.

    This bypasses MLXForwardSession.forward_via_kernel because that method
    hard-codes tier_index=0 in its _SyntheticOp construction. We rebuild
    the same forward path here but with custom tier_index injection.
    """
    mx, _ = _import_mlx()
    session.kernel.reset_caches()
    hidden = session._embed(tokens)

    for layer_id in range(session.num_layers):
        for kind in ("attention", "mlp_dense"):
            op_id = f"layer_{layer_id}.{kind}"
            tier = tier_overrides.get(op_id, 0)
            op = _RuntimeOp(op_id, layer_id, kind, tier)
            hidden = session.kernel.execute(op, decision=None, hidden=hidden)

    hidden = session._final_norm(hidden)
    logits = session._lm_head(hidden)
    mx.eval(logits)
    return logits


def run_test_bc1(model_id: str, prompt: str, verbose: bool) -> int:
    """
    The kernel→bank control loop test.

    Setup:
        - 2-tier plan: tier 0 = 4-bit, tier 1 = fp16.
        - Bank built against this plan (every op initialized to tier 0).
        - Session with the bank attached.

    Run:
        Forward A: every op at tier 0 (all 4-bit).
        Forward B: layer_0.attention at tier 1 (fp16), every other op at
                   tier 0. Driven through tier_index on the ScheduledOp,
                   which the kernel sees and forwards to bank.swap.

    Oracle:
        Load a fresh model. Quantize EVERY layer to 4-bit in place EXCEPT
        layer_0.self_attn (left as fp16). This matches what Forward B
        should produce numerically.

    Pass criterion:
        (1) A != B (swap actually changed math): max_abs_diff(A, B) > 0.01
        (2) B matches the oracle:
                cosine(B, oracle) >= 0.99999
                max_abs_diff(B, oracle) < 0.01

    If (1) fails: the kernel ignored tier_index or the bank silently failed.
    If (2) fails: the swap installed wrong weights (numerical bug).
    """
    mx, mlx_lm = _import_mlx()

    print("=" * 70)
    print("TEST BC1: kernel ↔ bank control loop")
    print("=" * 70)
    print(f"Model:  {model_id}")
    print(f"Prompt: {prompt!r}")
    print()

    # ------------------------------------------------------------------
    # Build A: bank-driven, all ops at tier 0 (4-bit).
    # ------------------------------------------------------------------
    print("Loading model A (bank-driven, mixed via tier_index)...")
    model_a, tokenizer = mlx_lm.load(model_id)
    tokens_list = tokenizer.encode(prompt)
    if hasattr(tokens_list, "tolist"):
        tokens_list = tokens_list.tolist()
    tokens = mx.array([tokens_list])

    from substrate.backend.mlx_session import MLXForwardSession
    session_a = MLXForwardSession(model_a)
    layers_a = session_a.kernel._layers

    # Two-tier plan: tier 0 = 4bit (default), tier 1 = fp16 (escalation).
    # The IR requires monotonically non-decreasing peak_ram_delta_bytes,
    # which is naturally satisfied because fp16 is more bits than 4bit.
    from scripts.test_weight_bank import build_test_plan, build_op_tier_precisions
    plan = build_test_plan(model_id, layers_a, {0: "4bit", 1: "fp16_eq"})
    precisions = build_op_tier_precisions(plan, {0: "4bit", 1: "fp16_eq"})

    print("Building store with both tiers...")
    from substrate.backend.ram_weight_store import RAMWeightStore
    from substrate.backend.weight_bank import WeightBank
    ops_with_tiers = {ob.op_id: [0, 1] for ob in plan.op_bundles}
    store = RAMWeightStore(model_a, layers_a, ops_with_tiers, precisions)
    bank = WeightBank(model_a, layers_a, store, plan)
    print(f"  Bank ready, all {len(plan.op_bundles)} ops at tier 0 (4-bit)")

    # Attach the bank to the session — the kernel will now consult it
    # on every execute() call.
    session_a.attach_weight_bank(bank)
    print(f"  Session attached to bank.")
    print()

    # Forward A: every op at tier 0.
    print("Forward A: every op at tier 0 (all 4-bit)...")
    logits_a = forward_with_tier_overrides(session_a, tokens, tier_overrides={})
    last_a = extract_last_logits(logits_a)
    print(f"  top-1: {max(range(len(last_a)), key=lambda i: last_a[i])}")
    print(f"  bank state sample: {dict(list(bank.active_state().items())[:3])}")
    print()

    # Forward B: layer_0.attention pinned to tier 1 (fp16). Rest stay at tier 0.
    print("Forward B: layer_0.attention at tier 1 (fp16), rest at tier 0...")
    logits_b = forward_with_tier_overrides(
        session_a, tokens, tier_overrides={"layer_0.attention": 1},
    )
    last_b = extract_last_logits(logits_b)
    print(f"  top-1: {max(range(len(last_b)), key=lambda i: last_b[i])}")
    print(f"  bank state sample: {dict(list(bank.active_state().items())[:3])}")
    print(f"  layer_0.attention active tier: {bank.active_tier('layer_0.attention')}")
    print()

    # ------------------------------------------------------------------
    # Build oracle: fresh model with everything 4-bit EXCEPT layer_0 attn.
    # ------------------------------------------------------------------
    print("Loading oracle model (in-place quantize all-but-one)...")
    model_oracle, _ = mlx_lm.load(model_id)
    session_oracle = MLXForwardSession(model_oracle)
    layers_oracle = session_oracle.kernel._layers

    from substrate.backend.quantize import quantize_module_in_place
    total_modified = 0
    for layer_id in range(len(layers_oracle)):
        # Skip layer_0.self_attn (left as fp16).
        if layer_id == 0:
            total_modified += quantize_module_in_place(layers_oracle[0].mlp, bits=4)
            continue
        total_modified += quantize_module_in_place(layers_oracle[layer_id].self_attn, bits=4)
        total_modified += quantize_module_in_place(layers_oracle[layer_id].mlp, bits=4)
    print(f"  Quantized {total_modified} weight arrays (all except layer_0.self_attn)")

    print("Running oracle forward...")
    session_oracle.reset()
    logits_oracle = session_oracle.forward_via_kernel(tokens)
    last_oracle = extract_last_logits(logits_oracle)
    print(f"  top-1: {max(range(len(last_oracle)), key=lambda i: last_oracle[i])}")
    print()

    # ------------------------------------------------------------------
    # Comparisons.
    # ------------------------------------------------------------------
    print("Comparisons:")
    cos_ab = cosine_similarity(last_a, last_b)
    diff_ab = max_abs_diff(last_a, last_b)
    print(f"  A vs B (must DIFFER):     cosine={cos_ab:.10f}, max_abs_diff={diff_ab:.6e}")

    cos_b_oracle = cosine_similarity(last_b, last_oracle)
    diff_b_oracle = max_abs_diff(last_b, last_oracle)
    print(f"  B vs oracle (must MATCH): cosine={cos_b_oracle:.10f}, max_abs_diff={diff_b_oracle:.6e}")

    cos_a_oracle = cosine_similarity(last_a, last_oracle)
    diff_a_oracle = max_abs_diff(last_a, last_oracle)
    print(f"  A vs oracle (control):    cosine={cos_a_oracle:.10f}, max_abs_diff={diff_a_oracle:.6e}")
    print()

    # Pass:
    #   (1) A != B: cosine < 0.99 OR max_abs_diff > 0.05 (some real divergence)
    #   (2) B == oracle: cosine >= 0.99999, max_abs_diff < 0.01
    passed = True
    failures = []
    if cos_ab >= 0.9999 and diff_ab < 0.01:
        passed = False
        failures.append(
            f"A == B (cosine {cos_ab:.6f}, max_diff {diff_ab:.6f}); "
            "the swap had no effect — kernel/bank wiring is broken."
        )
    if cos_b_oracle < 0.99999 or diff_b_oracle > 0.01:
        passed = False
        failures.append(
            f"B != oracle (cosine {cos_b_oracle:.6f}, max_diff {diff_b_oracle:.6f}); "
            "the swap installed wrong weights — numerical bug."
        )

    print()
    if passed:
        print("✓ TEST BC1 PASSED")
        print("  The kernel→bank control loop works:")
        print("    (1) Changing tier_index produced different logits.")
        print(f"        A != B with max_abs_diff={diff_ab:.4f}")
        print("    (2) The new logits match the in-place-quantized oracle.")
        print(f"        B matches oracle within max_abs_diff={diff_b_oracle:.6e}")
        return 0
    print("✗ TEST BC1 FAILED")
    for f in failures:
        print(f"  - {f}")
    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16",
    )
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    return run_test_bc1(args.model, args.prompt, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
