"""
substrate.bench.plan_builders — Hand-built PlanBundles for tests and benchmarks.

The real RDO solver in substrate.compiler.planner is great when you have
a calibration table and want a quality-cost-optimal plan. It is overkill
when you just want "every op admits these specific tiers, fixed precision
per tier, here is a PlanBundle the bank/kernel/controller can consume."

These builders synthesize the minimum that the IR validator demands:
- TensorMetadata catalog with skeleton + residuals so requires-edges
  resolve.
- ScheduledOps with monotonically non-decreasing peak_ram_delta_bytes
  across tiers (the IR's escalation invariant).
- Quality risk that lowers as tier increases (higher precision = lower
  risk).
- A Budget large enough that feasibility checks always pass.

Tier 0 is always the LOWEST-precision (cheapest) tier per the IR's
convention; tier N>0 escalates upward. So `tier_precisions={0: '4bit',
1: '6bit', 2: 'fp16_eq'}` produces a plan where:

    bank starts every op at 4-bit
    swap to tier 1 → 6-bit
    swap to tier 2 → fp16

If you pass `{0: 'fp16_eq', 1: '4bit'}`, the builder still sorts by
precision and emits tier 0 = 4bit, tier 1 = fp16. The keys are
ignored except for set membership; what matters is the set of
precisions you want admitted.
"""

from __future__ import annotations

from typing import Any


# Precision -> bit count. fp16_eq is treated as 16-bit for sorting.
_BITS_ORDER: dict[str, int] = {
    "2bit": 2,
    "3bit": 3,
    "4bit": 4,
    "6bit": 6,
    "fp16_eq": 16,
}


def build_test_plan(
    model_id: str,
    layers: list,
    tier_precisions: dict[int, str],
    sensitive_layers: tuple[int, ...] = (0, 1),
):
    """
    Build a tiny Substrate plan with hand-picked tier admissions.

    Args:
        model_id: Free-form string for the plan's model_id field. Used
                  by the bank for diagnostics; doesn't have to match a
                  real HuggingFace model ID for tests.
        layers: list of transformer block modules (kernel._layers).
                Used to count attention/mlp parameters per layer.
        tier_precisions: dict whose VALUES are the set of precisions to
                         admit. Keys are ignored. Example:
                            {0: '4bit', 1: '6bit', 2: 'fp16_eq'}
                         produces a 3-tier ladder sorted by bits ascending.
        sensitive_layers: layer indices treated as higher-risk for the
                          synthetic quality risk values. Defaults to
                          (0, 1) — first two layers, which are usually
                          the most quantization-sensitive.

    Returns:
        PlanBundle that the IR validator accepts and that
        WeightBank/MLXOpKernel/TierController can drive.

    The returned plan is NOT solver-optimal. It's a fixture that exposes
    a stable API surface for tests and benchmarks.
    """
    from substrate.compiler.ir import (
        Budget, EscalationPolicy, FallbackPolicy, FallbackStrategy, OpBundle,
        OpKind, PlanBundle, ScheduledOp, TensorMetadata, TensorRef,
    )
    from substrate.backend.quantize import snapshot_weights

    # Param count per op kind. Use layer 0 as canonical; assume all
    # layers have the same shape (true for non-MoE, dense models).
    attn0 = layers[0].self_attn
    mlp0 = layers[0].mlp

    def count(module) -> int:
        snaps = snapshot_weights(module)
        total = 0
        for _, _, arr in snaps:
            n = 1
            for d in arr.shape:
                n *= int(d)
            total += n
        return total

    attn_params = count(attn0)
    mlp_params = count(mlp0)

    # Tier ordering: by bits ascending. Tier 0 = cheapest, tier N = most expensive.
    # This matches the IR's invariant that peak_ram_delta_bytes monotonically
    # increases with tier_index.
    tiers_sorted = sorted(tier_precisions.items(), key=lambda kv: _BITS_ORDER[kv[1]])

    catalog: dict[str, TensorMetadata] = {}
    op_bundles: list[OpBundle] = []

    for layer_id in range(len(layers)):
        for kind, params, op_kind_enum in [
            ("attention", attn_params, OpKind.ATTENTION),
            ("mlp_dense", mlp_params, OpKind.MLP_DENSE),
        ]:
            op_id = f"layer_{layer_id}.{kind}"

            # Skeleton: tier-0 weights stored at lowest-precision equivalent.
            # bytes ≈ params * tier0_bits / 8.
            tier0_bits = _BITS_ORDER[tiers_sorted[0][1]]
            sk_id = f"{op_id}.skeleton"
            sk_bytes = int(params * tier0_bits / 8)
            catalog[sk_id] = TensorMetadata(
                tensor_id=sk_id, bytes_in_ram=sk_bytes,
                bytes_on_ssd=sk_bytes, is_skeleton=True,
                layer_id=layer_id, tier_index=0,
            )

            tiers: list[ScheduledOp] = []
            prev_bytes = sk_bytes
            for new_tier_idx, (_orig_key, prec) in enumerate(tiers_sorted):
                bits = _BITS_ORDER[prec]
                full_bytes = int(params * bits / 8)

                # Residual deltas: tier N>0 needs the bytes between this
                # tier and the previous one. Catalogued so that the
                # ScheduledOp's `requires` chain terminates in valid IDs.
                if new_tier_idx > 0:
                    delta_bytes = max(0, full_bytes - prev_bytes)
                    rid = f"{op_id}.residual_{new_tier_idx}"
                    if delta_bytes > 0 and rid not in catalog:
                        catalog[rid] = TensorMetadata(
                            tensor_id=rid, bytes_in_ram=delta_bytes,
                            bytes_on_ssd=delta_bytes, is_skeleton=False,
                            layer_id=layer_id, tier_index=new_tier_idx,
                        )

                # `requires`: skeleton + all residuals up to this tier.
                requires = [TensorRef(sk_id)]
                for t in range(1, new_tier_idx + 1):
                    rid = f"{op_id}.residual_{t}"
                    if rid in catalog:
                        requires.append(TensorRef(rid))

                # Quality risk: nominal sensitivity for first/last layers,
                # lower for middle layers. Decreasing in tier (higher precision
                # = lower risk).
                base_risk = 0.7 if layer_id in sensitive_layers else 0.3
                risk = max(0.0, base_risk / max(1, new_tier_idx + 1))

                # peak_ram_delta_bytes monotonic across tiers. Tier 0 = 0;
                # tier N>0 = full_bytes - tier0_bytes (the extra needed beyond
                # the skeleton).
                ram_delta = (
                    0 if new_tier_idx == 0
                    else max(0, full_bytes - sk_bytes)
                )

                tiers.append(ScheduledOp(
                    op_id=op_id,
                    tier_index=new_tier_idx,
                    op_kind=op_kind_enum,
                    layer_id=layer_id,
                    requires=tuple(requires),
                    prefetch=(),
                    evict_after=(),
                    fallback=FallbackStrategy.SKELETON_ONLY,
                    estimated_compute_us=200,
                    estimated_quality_risk=risk,
                    peak_ram_delta_bytes=ram_delta,
                ))
                prev_bytes = full_bytes

            op_bundles.append(OpBundle(op_id=op_id, tiers=tuple(tiers)))

    # Budget large enough that feasibility checks pass trivially. The
    # builder is for tests/benchmarks; budget tuning is the real solver's job.
    peak_ram = (
        sum(meta.bytes_in_ram for meta in catalog.values() if meta.is_skeleton)
        + 2_000_000_000   # rough fixed overhead for embeddings/lm_head/etc.
    )
    pool = 4_000_000_000
    budget = Budget(
        max_ram_bytes=peak_ram + pool,
        max_ssd_cache_bytes=int(50e9),
        sustained_ssd_bw_bytes_per_sec=int(5e9),
        quality_loss_cap=0.5,
        target_tokens_per_second=30.0,
    )

    return PlanBundle(
        model_id=model_id,
        budget=budget,
        tensor_catalog=catalog,
        op_bundles=tuple(op_bundles),
        escalation_policy=EscalationPolicy(),
        fallback_policy=FallbackPolicy(),
        predicted_peak_resident_bytes=peak_ram,
        predicted_steady_state_resident_bytes=peak_ram,
        predicted_ssd_bandwidth_bps=0,
        predicted_tokens_per_second=30.0,
        predicted_quality_loss=0.05,
        escalation_ram_pool_bytes=pool,
        solver_version="bench-fixture",
        solver_notes=("hand-built fixture, not from the real solver",),
    )


def build_op_tier_precisions(
    plan,
    tier_precisions: dict[int, str],
) -> dict[tuple[str, int], str]:
    """
    Build the (op_id, tier_index) -> precision_string map that
    RAMWeightStore consumes.

    The plan was built with tiers sorted by bits ascending; this map
    must match that ordering. The keys of `tier_precisions` are ignored;
    only the set of precisions matters.
    """
    sorted_precisions = sorted(
        tier_precisions.values(), key=lambda p: _BITS_ORDER[p],
    )
    out: dict[tuple[str, int], str] = {}
    for ob in plan.op_bundles:
        for new_tier_idx, prec in enumerate(sorted_precisions):
            out[(ob.op_id, new_tier_idx)] = prec
    return out
