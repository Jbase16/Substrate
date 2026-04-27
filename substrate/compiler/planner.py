"""
substrate.compiler.planner — Compile a ModelProfile + Budget into a PlanBundle.

The planner produces a maneuver envelope: a default execution path plus
pre-validated escalation tiers per op. Quality estimation is centralized in
HybridQualityEstimator; there is no separate inline surrogate. When no real
calibration data is available, the planner builds a stub calibration table
and feeds it to the same estimator, with a loud solver_note marking the run
as ungrounded.

Pipeline:
    1. Feasibility check. Hard abort with structured report on failure.
    2. Greedy rate-distortion fill: pick tier-0 precision per op by querying
       HybridQualityEstimator for each candidate upgrade's marginal loss.
    3. Build escalation precisions (tier-1 = +1 step, tier-2 = +2 steps).
    4. Build tensor catalog covering all (op, precision) pairs we may use.
    5. Assign residency: skeleton always resident; residuals fit into RAM
       headroom in sensitivity order; remainder streams.
    6. Compute the escalation pool (= max_ram - default-path peak resident).
    7. Admit tier-1 / tier-2 escalations to the pool: K-largest-delta
       constraint enforced here.
    8. Build OpBundles with prefetch / evict / fallback annotations.
    9. Emit PlanBundle. The IR validates structural invariants on
       construction; this module guarantees they hold.

Design note on quality estimator queries:
    For each candidate upgrade, we build a full (layer, op_kind) -> Precision
    assignment dict and call estimator.estimate() to get the marginal loss.
    This costs O(N) per query where N is the number of ops. Total work in
    the fill loop is O(iters * candidates * N) = O(N^3) in the worst case,
    which for ~80 ops is ~500k integer dict lookups. Negligible at compile
    time; we are not optimizing solver speed here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from substrate.compiler.feasibility import (
    BindingAxis,
    FeasibilityReport,
    InfeasibleBudgetError,
    check_feasibility,
)
from substrate.compiler.ir import (
    Budget,
    EscalationPolicy,
    EvictRule,
    FallbackPolicy,
    FallbackStrategy,
    OpBundle,
    OpKind,
    PlanBundle,
    PrefetchRequest,
    ScheduledOp,
    TensorMetadata,
    TensorRef,
)
from substrate.compiler.quality import (
    HybridQualityEstimator,
    QualityEstimator,
    stub_calibration_table,
)

log = logging.getLogger(__name__)
SOLVER_VERSION = "0.2.0"


# ---------------------------------------------------------------------------
# Precision tiers (the planner's internal axis).
# ---------------------------------------------------------------------------
class Precision(str, Enum):
    SKELETON_2BIT = "2bit"
    SKELETON_PLUS_R1 = "3bit"
    SKELETON_PLUS_R2 = "4bit"
    REFINED_6BIT = "6bit"
    NEAR_FP16 = "fp16_eq"

    @property
    def effective_bits(self) -> float:
        return {"2bit": 2.0, "3bit": 3.0, "4bit": 4.0, "6bit": 6.0, "fp16_eq": 16.0}[self.value]


_PRECISION_LADDER: tuple[Precision, ...] = (
    Precision.SKELETON_2BIT,
    Precision.SKELETON_PLUS_R1,
    Precision.SKELETON_PLUS_R2,
    Precision.REFINED_6BIT,
    Precision.NEAR_FP16,
)


# ---------------------------------------------------------------------------
# Profile types.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpProfile:
    op_id: str
    op_kind: OpKind
    layer_id: int
    param_count: int
    skeleton_compute_us: int
    full_precision_compute_us: int
    sensitivity: float
    minimum_residual_bytes_per_token: int = 0
    moe_top_k: int = 0
    moe_num_experts: int = 0

    @property
    def skeleton_bytes(self) -> int:
        return int(self.param_count * Precision.SKELETON_2BIT.effective_bits / 8)

    def bytes_at_precision(self, p: Precision) -> int:
        return int(self.param_count * p.effective_bits / 8)

    @property
    def best_quality_loss(self) -> float:
        return 0.0001


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    ops: tuple[OpProfile, ...]
    embedding_bytes: int
    lm_head_bytes: int
    runtime_overhead_bytes: int


# ---------------------------------------------------------------------------
# Planner options.
# ---------------------------------------------------------------------------
@dataclass
class PlannerOptions:
    max_upgrade_iterations: int = 2000
    moe_prefetch_top_k: int = 4
    emit_escalation_tiers: bool = True


# ---------------------------------------------------------------------------
# The planner.
# ---------------------------------------------------------------------------
class Planner:
    """
    Compiles a ModelProfile + Budget into a PlanBundle.

    Quality estimation is centralized in HybridQualityEstimator. If no
    estimator is provided to __init__, the planner builds one from a stub
    calibration table at compile time and tags the run with a loud
    solver_note. There is NO separate surrogate code path — the estimator
    is the single source of truth for quality math.

    The estimator's assignment dict is keyed by (layer_id, op_kind). This
    means architectures with multiple ops of the same op_kind in the same
    layer (e.g., two attention blocks per layer) collapse into one entry.
    For Qwen/Llama-style architectures this is fine; for hypothetical
    multi-attention models, see _build_assignment_for_estimator.
    """

    def __init__(
        self,
        quality_estimator: QualityEstimator | None = None,
        opts: PlannerOptions | None = None,
        escalation_policy: EscalationPolicy | None = None,
        fallback_policy: FallbackPolicy | None = None,
    ) -> None:
        self.quality = quality_estimator
        self.opts = opts or PlannerOptions()
        self.escalation_policy = escalation_policy or EscalationPolicy()
        self.fallback_policy = fallback_policy or FallbackPolicy()

    # ------------------------------------------------------------------
    # Entry point.
    # ------------------------------------------------------------------
    def compile(self, profile: ModelProfile, budget: Budget) -> PlanBundle:
        notes: list[str] = []

        # 1. Feasibility.
        report = check_feasibility(profile, budget)
        if not report.feasible:
            raise InfeasibleBudgetError(report)
        notes.append(
            f"feasibility ok: floor_ram={report.floor_resident_bytes / 1e9:.2f}GB"
        )

        # 2. Resolve the quality estimator. If the user didn't provide one,
        #    build a stub table and feed it into HybridQualityEstimator. This
        #    keeps a single quality code path; the only difference between
        #    "real calibration" and "stub" is the table contents.
        estimator, grounding = self._resolve_estimator(profile, notes)

        # 3. Greedy RDO precision assignment for tier 0.
        tier0_precision = self._greedy_quality_fill(
            profile, budget, estimator, notes,
        )

        # 4. Decide tier-1 / tier-2 precision per op.
        if self.opts.emit_escalation_tiers:
            tier1_precision, tier2_precision = self._build_escalation_precisions(
                profile, tier0_precision,
            )
        else:
            tier1_precision = {}
            tier2_precision = {}

        # 5. Build the catalog covering every (op, precision) we may use.
        catalog = self._build_catalog(
            profile, tier0_precision, tier1_precision, tier2_precision,
        )

        # 6. Assign residency for tier-0.
        residency = self._assign_residency(
            profile, budget, catalog, tier0_precision, notes,
        )

        # 7. Compute pool size and admit ops to escalation tiers.
        steady_ram, peak_ram = self._compute_ram_envelope(
            profile, catalog, residency,
        )
        pool = max(0, budget.max_ram_bytes - peak_ram)
        notes.append(
            f"escalation_pool: {pool / 1e6:.1f}MB "
            f"(peak_resident={peak_ram / 1e9:.2f}GB, "
            f"max_ram={budget.max_ram_bytes / 1e9:.2f}GB)"
        )
        admitted_tier1, admitted_tier2 = self._admit_escalations_to_pool(
            profile, tier0_precision, tier1_precision, tier2_precision,
            pool, notes,
        )

        # 8. Build OpBundles.
        op_bundles = self._build_bundles(
            profile, catalog, residency,
            tier0_precision, admitted_tier1, admitted_tier2,
        )

        # 9. Aggregate metrics for the default path. Quality loss comes from
        #    the SAME estimator that drove the fill. No second source of truth.
        ssd_bw = self._compute_ssd_bandwidth(profile, catalog, residency, budget)
        compute_us = sum(ob.default.estimated_compute_us for ob in op_bundles)
        tps = 1e6 / max(1, compute_us)
        quality_estimate = estimator.estimate(
            self._build_assignment_for_estimator(profile, tier0_precision)
        )
        quality_loss = quality_estimate.expected_loss
        notes.append(
            f"quality estimate ({grounding}): loss={quality_loss:.4f} "
            f"binding_layers={list(quality_estimate.binding_layers)}"
        )

        # 10. Quality-axis feasibility. If the fill couldn't reach the cap,
        #     surface this as a structured QUALITY infeasibility instead of
        #     letting PlanBundle's validator raise. The user gets a clean
        #     error with relax_options and the binding layers identified.
        if quality_loss > budget.quality_loss_cap:
            raise InfeasibleBudgetError(FeasibilityReport(
                feasible=False,
                binding_axis=BindingAxis.QUALITY,
                reason=(
                    f"Greedy RDO fill could not reach the quality cap "
                    f"({budget.quality_loss_cap:.4f}) given the available "
                    f"{grounding} calibration. Best achievable loss: "
                    f"{quality_loss:.4f}. Binding layers: "
                    f"{list(quality_estimate.binding_layers)}."
                ),
                relax_options={
                    "quality_loss_cap": f">= {quality_loss:.4f}",
                    "max_ram_bytes": (
                        "increase to allow higher precision on more ops "
                        f"(current peak: {peak_ram} bytes)"
                    ),
                    "model": (
                        "use a model with better quantization characteristics "
                        "for the binding layers"
                    ),
                },
                ceiling_quality_loss=quality_loss,
            ))

        return PlanBundle(
            model_id=profile.model_id,
            budget=budget,
            tensor_catalog=catalog,
            op_bundles=tuple(op_bundles),
            escalation_policy=self.escalation_policy,
            fallback_policy=self.fallback_policy,
            predicted_peak_resident_bytes=peak_ram,
            predicted_steady_state_resident_bytes=steady_ram,
            predicted_ssd_bandwidth_bps=ssd_bw,
            predicted_tokens_per_second=tps,
            predicted_quality_loss=quality_loss,
            escalation_ram_pool_bytes=pool,
            solver_version=SOLVER_VERSION,
            solver_notes=tuple(notes),
        )

    # ------------------------------------------------------------------
    # Estimator resolution. Single quality code path.
    # ------------------------------------------------------------------
    def _resolve_estimator(
        self, profile: ModelProfile, notes: list[str],
    ) -> tuple[QualityEstimator, str]:
        """
        Returns (estimator, grounding_label) where grounding_label is one of:
            'calibrated' — user provided a real estimator
            'stub'       — we built one from stub_calibration_table

        The grounding label is included in solver_notes so consumers of the
        compiled plan know whether the predicted quality is grounded in
        measured data or fabricated.
        """
        if self.quality is not None:
            notes.append(
                "quality_grounding=calibrated "
                "(estimator provided by caller; predictions reflect measured calibration)"
            )
            return self.quality, "calibrated"

        # Build a stub table covering the op kinds and precisions we'll use.
        op_kinds = tuple({op.op_kind for op in profile.ops})
        sensitive_layers = tuple({
            op.layer_id for op in profile.ops if op.sensitivity >= 0.6
        })
        num_layers = max((op.layer_id for op in profile.ops), default=-1) + 1
        table = stub_calibration_table(
            num_layers=num_layers,
            op_kinds=op_kinds,
            precisions=_PRECISION_LADDER,
            sensitive_layers=sensitive_layers,
        )
        notes.append(
            "quality_grounding=stub "
            "(NO real calibration data; predictions are fabricated. "
            "Run scripts/calibrate.py and pass --calibration to ground them.)"
        )
        log.warning(
            "Planner.compile: no quality_estimator provided; using stub table. "
            "Predicted quality loss will not reflect actual model behavior."
        )
        return HybridQualityEstimator(table), "stub"

    @staticmethod
    def _build_assignment_for_estimator(
        profile: ModelProfile, precision_map: Mapping[str, Precision],
    ) -> dict[tuple[int, OpKind], Precision]:
        """
        Convert the planner's op_id -> Precision map into the (layer, op_kind)
        -> Precision shape that HybridQualityEstimator wants.

        For models where the same (layer_id, op_kind) appears multiple times
        (rare; not in Qwen/Llama), the LAST entry wins. This is acceptable
        for v0.1 because the estimator is keyed by layer×kind anyway and
        such collisions cannot be distinguished without extending the
        estimator's key space.
        """
        out: dict[tuple[int, OpKind], Precision] = {}
        for op in profile.ops:
            out[(op.layer_id, op.op_kind)] = precision_map[op.op_id]
        return out

    # ------------------------------------------------------------------
    # Greedy RDO fill.
    # ------------------------------------------------------------------
    def _greedy_quality_fill(
        self,
        profile: ModelProfile,
        budget: Budget,
        estimator: QualityEstimator,
        notes: list[str],
    ) -> dict[str, Precision]:
        """
        Iteratively promote the op with the best (Δquality / ΔRAM) ratio
        until either the quality cap is met or no profitable upgrade fits
        within the RAM budget.

        On each iteration:
            - Build the current (layer, op_kind) -> Precision assignment.
            - Get current loss from the estimator.
            - For each op that can be upgraded by one precision step:
                * Build a candidate assignment with that swap.
                * Get candidate loss from the estimator.
                * Compute Δloss and ΔRAM.
                * Score = Δloss / ΔRAM (no extra weighting; the estimator
                  already encodes per-layer sensitivity via its calibration
                  table — we don't need to double-count it here).
            - Pick the highest-scoring candidate; commit; repeat.

        Convergence target: total loss <= budget.quality_loss_cap * 0.95.
        The 5% slack avoids over-upgrading once we're near the cap, which
        wastes RAM on diminishing returns.
        """
        precision: dict[str, Precision] = {
            op.op_id: Precision.SKELETON_2BIT for op in profile.ops
        }
        op_by_id = {op.op_id: op for op in profile.ops}

        def total_ram(p_map: Mapping[str, Precision]) -> int:
            base = (
                profile.embedding_bytes
                + profile.lm_head_bytes
                + profile.runtime_overhead_bytes
            )
            return base + sum(
                op_by_id[oid].bytes_at_precision(p) for oid, p in p_map.items()
            )

        def loss_for(p_map: Mapping[str, Precision]) -> float:
            """
            Sum of per-layer losses, UNCLAMPED. Using the estimator's
            expected_loss (which clamps at 1.0) makes the fill loop blind
            to gradient when many ops are at low precision and the total
            saturates. Per-layer loss is monotonic and unsaturated, so it
            preserves the marginal signal the fill loop needs.

            The clamped value is what downstream consumers see in the final
            PlanBundle.predicted_quality_loss — that's correct because once
            the saturated bound is reached, the model is qualitatively bad
            regardless of which 1.0+ value the unclamped sum hits.
            """
            assignment = self._build_assignment_for_estimator(profile, p_map)
            estimate = estimator.estimate(assignment)
            return sum(estimate.per_layer_loss.values())

        # Convergence target uses the clamped expected_loss (what the user's
        # cap is in). Upgrade scoring uses the unclamped per-layer sum (so
        # we have gradient signal when totals saturate at 1.0).
        def clamped_loss_for(p_map: Mapping[str, Precision]) -> float:
            assignment = self._build_assignment_for_estimator(profile, p_map)
            return estimator.estimate(assignment).expected_loss

        target = budget.quality_loss_cap * 0.95
        initial_clamped = clamped_loss_for(precision)
        initial_unclamped = loss_for(precision)
        notes.append(
            f"tier-0 fill: initial clamped_loss={initial_clamped:.4f} "
            f"unclamped_sum={initial_unclamped:.4f} target={target:.4f}"
        )

        for it in range(self.opts.max_upgrade_iterations):
            current_clamped = clamped_loss_for(precision)
            if current_clamped <= target:
                notes.append(
                    f"tier-0 fill converged at iter {it} "
                    f"clamped_loss={current_clamped:.4f}"
                )
                return precision
            current_loss = loss_for(precision)

            current_ram = total_ram(precision)
            best_op_id: str | None = None
            best_target: Precision | None = None
            best_score = 0.0

            for op_id, cur_p in precision.items():
                idx = _PRECISION_LADDER.index(cur_p)
                if idx + 1 >= len(_PRECISION_LADDER):
                    continue
                nxt = _PRECISION_LADDER[idx + 1]
                d_ram = (
                    op_by_id[op_id].bytes_at_precision(nxt)
                    - op_by_id[op_id].bytes_at_precision(cur_p)
                )
                if d_ram <= 0 or current_ram + d_ram > budget.max_ram_bytes:
                    continue

                # Marginal estimator query: full assignment with one swap.
                candidate = dict(precision)
                candidate[op_id] = nxt
                cand_loss = loss_for(candidate)
                d_loss = current_loss - cand_loss
                if d_loss <= 0:
                    continue

                # Pure rate-distortion ratio. The estimator already accounts
                # for layer sensitivity through its calibration table, so we
                # don't multiply by op.sensitivity here (that would double-
                # count and bias toward layers the calibration already
                # marked sensitive).
                score = d_loss / d_ram
                if score > best_score:
                    best_score = score
                    best_op_id = op_id
                    best_target = nxt

            if best_op_id is None or best_target is None:
                notes.append(
                    f"tier-0 fill stuck at iter {it} loss={current_loss:.4f} "
                    f"(no profitable upgrade fits RAM budget)"
                )
                return precision
            precision[best_op_id] = best_target

        notes.append(
            f"tier-0 fill: max_upgrade_iterations exhausted "
            f"(loss={loss_for(precision):.4f})"
        )
        return precision

    # ------------------------------------------------------------------
    # Escalation precision proposals (precision-only ladder).
    # ------------------------------------------------------------------
    @staticmethod
    def _build_escalation_precisions(
        profile: ModelProfile, tier0: Mapping[str, Precision],
    ) -> tuple[dict[str, Precision], dict[str, Precision]]:
        tier1: dict[str, Precision] = {}
        tier2: dict[str, Precision] = {}
        for op in profile.ops:
            cur = tier0[op.op_id]
            idx = _PRECISION_LADDER.index(cur)
            if idx + 1 < len(_PRECISION_LADDER):
                tier1[op.op_id] = _PRECISION_LADDER[idx + 1]
            if idx + 2 < len(_PRECISION_LADDER):
                tier2[op.op_id] = _PRECISION_LADDER[idx + 2]
        return tier1, tier2

    # ------------------------------------------------------------------
    # Catalog covering every (op, precision) pair we may reference.
    # ------------------------------------------------------------------
    @staticmethod
    def _build_catalog(
        profile: ModelProfile,
        tier0: Mapping[str, Precision],
        tier1: Mapping[str, Precision],
        tier2: Mapping[str, Precision],
    ) -> dict[str, TensorMetadata]:
        catalog: dict[str, TensorMetadata] = {}
        for op in profile.ops:
            highest = tier0[op.op_id]
            for proposed in (tier1.get(op.op_id), tier2.get(op.op_id)):
                if proposed is not None and (
                    _PRECISION_LADDER.index(proposed)
                    > _PRECISION_LADDER.index(highest)
                ):
                    highest = proposed
            highest_idx = _PRECISION_LADDER.index(highest)

            sk_id = f"{op.op_id}.skeleton"
            sk_bytes = op.bytes_at_precision(Precision.SKELETON_2BIT)
            catalog[sk_id] = TensorMetadata(
                tensor_id=sk_id,
                bytes_in_ram=sk_bytes,
                bytes_on_ssd=sk_bytes,
                is_skeleton=True,
                layer_id=op.layer_id,
                tier_index=0,
            )
            for t in range(1, highest_idx + 1):
                tier_prec = _PRECISION_LADDER[t]
                full_bytes = op.bytes_at_precision(tier_prec)
                base_bytes = op.bytes_at_precision(_PRECISION_LADDER[t - 1])
                residual_bytes = max(0, full_bytes - base_bytes)
                if residual_bytes == 0:
                    continue
                rid = f"{op.op_id}.residual_{t}"
                catalog[rid] = TensorMetadata(
                    tensor_id=rid,
                    bytes_in_ram=residual_bytes,
                    bytes_on_ssd=residual_bytes,
                    is_skeleton=False,
                    layer_id=op.layer_id,
                    tier_index=t,
                )
        return catalog

    # ------------------------------------------------------------------
    # Residency assignment for tier-0.
    # ------------------------------------------------------------------
    @staticmethod
    def _assign_residency(
        profile: ModelProfile,
        budget: Budget,
        catalog: Mapping[str, TensorMetadata],
        tier0: Mapping[str, Precision],
        notes: list[str],
    ) -> dict[str, str]:
        op_by_id = {op.op_id: op for op in profile.ops}
        residency: dict[str, str] = {}
        ram_used = (
            profile.embedding_bytes
            + profile.lm_head_bytes
            + profile.runtime_overhead_bytes
        )

        # Skeletons always resident.
        for tid, meta in catalog.items():
            if meta.is_skeleton:
                residency[tid] = "resident"
                ram_used += meta.bytes_in_ram

        # Partition residuals: those needed by tier-0 vs higher tiers.
        tier0_required: list[tuple[str, TensorMetadata]] = []
        higher_tier: list[tuple[str, TensorMetadata]] = []
        for tid, meta in catalog.items():
            if meta.is_skeleton:
                continue
            owner = tid.rsplit(".", 1)[0]
            owner_tier_idx = _PRECISION_LADDER.index(tier0[owner])
            if meta.tier_index <= owner_tier_idx:
                tier0_required.append((tid, meta))
            else:
                higher_tier.append((tid, meta))

        # Sort tier-0 residuals by sensitivity (high first), tier index ascending.
        def score(item: tuple[str, TensorMetadata]) -> tuple[float, int]:
            tid, meta = item
            owner = tid.rsplit(".", 1)[0]
            sens = op_by_id[owner].sensitivity if owner in op_by_id else 0.0
            return (-sens, meta.tier_index)

        tier0_required.sort(key=score)
        rc = sc = 0
        for tid, meta in tier0_required:
            if ram_used + meta.bytes_in_ram <= budget.max_ram_bytes:
                residency[tid] = "resident"
                ram_used += meta.bytes_in_ram
                rc += 1
            else:
                residency[tid] = "streamed"
                sc += 1

        # Higher-tier residuals always streamed (only loaded on escalation).
        for tid, _ in higher_tier:
            residency[tid] = "streamed"

        notes.append(
            f"residency: {rc} tier-0 residuals resident, "
            f"{sc} streamed, {len(higher_tier)} higher-tier streamed"
        )
        return residency

    # ------------------------------------------------------------------
    # Admit escalations to the pool.
    # ------------------------------------------------------------------
    def _admit_escalations_to_pool(
        self,
        profile: ModelProfile,
        tier0: Mapping[str, Precision],
        tier1: Mapping[str, Precision],
        tier2: Mapping[str, Precision],
        pool: int,
        notes: list[str],
    ) -> tuple[dict[str, Precision], dict[str, Precision]]:
        op_by_id = {op.op_id: op for op in profile.ops}
        K = self.escalation_policy.max_concurrent_escalations

        def delta(op_id: str, target: Precision) -> int:
            cur = tier0[op_id]
            return (
                op_by_id[op_id].bytes_at_precision(target)
                - op_by_id[op_id].bytes_at_precision(cur)
            )

        candidates = sorted(
            tier1.keys(),
            key=lambda oid: (-op_by_id[oid].sensitivity, delta(oid, tier1[oid])),
        )
        admitted_tier1: dict[str, Precision] = {}
        admitted_deltas: list[int] = []

        def top_k_sum(deltas: list[int]) -> int:
            return sum(deltas[:K]) if K > 0 else 0

        for op_id in candidates:
            d = delta(op_id, tier1[op_id])
            if d <= 0:
                admitted_tier1[op_id] = tier1[op_id]
                continue
            new_deltas = sorted(admitted_deltas + [d], reverse=True)
            if top_k_sum(new_deltas) <= pool:
                admitted_tier1[op_id] = tier1[op_id]
                admitted_deltas = new_deltas

        notes.append(
            f"tier-1 admission: {len(admitted_tier1)}/{len(tier1)} ops; "
            f"top-{K} delta sum = {top_k_sum(admitted_deltas) / 1e6:.1f}MB / "
            f"pool {pool / 1e6:.1f}MB"
        )

        admitted_tier2: dict[str, Precision] = {}
        admitted_t2_deltas: list[int] = []
        for op_id in candidates:
            if op_id not in admitted_tier1:
                continue
            t2 = tier2.get(op_id)
            if t2 is None:
                continue
            d = delta(op_id, t2)
            if d <= 0:
                admitted_tier2[op_id] = t2
                continue
            new_deltas = sorted(admitted_t2_deltas + [d], reverse=True)
            if top_k_sum(new_deltas) <= pool:
                admitted_tier2[op_id] = t2
                admitted_t2_deltas = new_deltas

        notes.append(
            f"tier-2 admission: {len(admitted_tier2)}/{len(tier2)} ops"
        )
        return admitted_tier1, admitted_tier2

    # ------------------------------------------------------------------
    # Build OpBundles.
    # ------------------------------------------------------------------
    def _build_bundles(
        self,
        profile: ModelProfile,
        catalog: Mapping[str, TensorMetadata],
        residency: Mapping[str, str],
        tier0: Mapping[str, Precision],
        tier1: Mapping[str, Precision],
        tier2: Mapping[str, Precision],
    ) -> list[OpBundle]:
        op_list = list(profile.ops)
        op_pos = {op.op_id: i for i, op in enumerate(op_list)}
        bundles: list[OpBundle] = []

        for op in op_list:
            tiers_list = [
                self._make_op_at_tier(
                    op, op_list, op_pos, catalog, residency,
                    tier0[op.op_id], tier_index=0,
                    base_precision=tier0[op.op_id],
                )
            ]
            if op.op_id in tier1:
                tiers_list.append(self._make_op_at_tier(
                    op, op_list, op_pos, catalog, residency,
                    tier1[op.op_id], tier_index=1,
                    base_precision=tier0[op.op_id],
                ))
            if op.op_id in tier2:
                tiers_list.append(self._make_op_at_tier(
                    op, op_list, op_pos, catalog, residency,
                    tier2[op.op_id], tier_index=2,
                    base_precision=tier0[op.op_id],
                ))
            bundles.append(OpBundle(op_id=op.op_id, tiers=tuple(tiers_list)))
        return bundles

    def _make_op_at_tier(
        self,
        op: OpProfile,
        op_list: list[OpProfile],
        op_pos: Mapping[str, int],
        catalog: Mapping[str, TensorMetadata],
        residency: Mapping[str, str],
        precision: Precision,
        tier_index: int,
        base_precision: Precision,
    ) -> ScheduledOp:
        prec_idx = _PRECISION_LADDER.index(precision)

        required: list[TensorRef] = [TensorRef(f"{op.op_id}.skeleton")]
        for t in range(1, prec_idx + 1):
            rid = f"{op.op_id}.residual_{t}"
            if rid in catalog:
                required.append(TensorRef(rid))

        prefetches: list[PrefetchRequest] = []
        my_pos = op_pos[op.op_id]
        prev_id = op_list[my_pos - 1].op_id if my_pos > 0 else None
        for tref in required:
            if residency[tref.tensor_id] == "streamed" and prev_id is not None:
                prefetches.append(PrefetchRequest(
                    tensor=tref,
                    start_during=prev_id,
                    deadline_before=op.op_id,
                    priority=1 + catalog[tref.tensor_id].tier_index,
                ))

        evicts: list[EvictRule] = [
            EvictRule(tensor=tref, after_op=op.op_id)
            for tref in required if residency[tref.tensor_id] == "streamed"
        ]

        fallback = (
            FallbackStrategy.SHARED_EXPERT
            if op.op_kind in (OpKind.MOE_EXPERT, OpKind.MOE_DISPATCH)
            else FallbackStrategy.SKELETON_ONLY
        )

        # Compute estimate: linear interpolation between skeleton/full.
        # Replace with measured costs from calibration once available.
        t_frac = prec_idx / (len(_PRECISION_LADDER) - 1)
        est_us = max(1, int(
            op.skeleton_compute_us
            + t_frac * (op.full_precision_compute_us - op.skeleton_compute_us)
        ))

        moe_likely: tuple[str, ...] = ()
        moe_top_k = 0
        if op.op_kind is OpKind.MOE_DISPATCH:
            moe_top_k = op.moe_top_k
            k = min(self.opts.moe_prefetch_top_k, op.moe_num_experts)
            moe_likely = tuple(f"{op.op_id}.expert_{j}" for j in range(k))

        # Per-op quality risk: this is a single-op marginal, used by the
        # tier_controller for tier ordering. We compute it here as a local
        # surrogate (1 / effective_bits, weighted by sensitivity) because
        # the global aggregate uses the estimator. Per-op risk is always
        # used in monotonic ordering across tiers within an op_bundle, so
        # the absolute value matters less than the ordering — and 1/bits
        # is monotonic in bits, which is what the IR validates.
        risk = min(1.0, 0.005 / precision.effective_bits * (1.0 + op.sensitivity))

        ram_delta = (
            0 if tier_index == 0
            else max(0, op.bytes_at_precision(precision) - op.bytes_at_precision(base_precision))
        )

        return ScheduledOp(
            op_id=op.op_id,
            tier_index=tier_index,
            op_kind=op.op_kind,
            layer_id=op.layer_id,
            requires=tuple(required),
            prefetch=tuple(prefetches),
            evict_after=tuple(evicts),
            fallback=fallback,
            estimated_compute_us=est_us,
            estimated_quality_risk=risk,
            peak_ram_delta_bytes=ram_delta,
            moe_likely_experts=moe_likely,
            moe_top_k=moe_top_k,
        )

    # ------------------------------------------------------------------
    # Aggregate metric helpers.
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_ram_envelope(
        profile: ModelProfile,
        catalog: Mapping[str, TensorMetadata],
        residency: Mapping[str, str],
    ) -> tuple[int, int]:
        steady = (
            profile.embedding_bytes
            + profile.lm_head_bytes
            + profile.runtime_overhead_bytes
            + sum(
                meta.bytes_in_ram
                for tid, meta in catalog.items()
                if residency[tid] == "resident"
            )
        )
        streamed = [
            meta.bytes_in_ram
            for tid, meta in catalog.items()
            if residency[tid] == "streamed"
        ]
        return steady, steady + (max(streamed) if streamed else 0)

    @staticmethod
    def _compute_ssd_bandwidth(
        profile: ModelProfile,
        catalog: Mapping[str, TensorMetadata],
        residency: Mapping[str, str],
        budget: Budget,
    ) -> int:
        per_token = sum(
            meta.bytes_on_ssd
            for tid, meta in catalog.items()
            if residency[tid] == "streamed"
        )
        return int(per_token * (budget.target_tokens_per_second or 1.0))


# ---------------------------------------------------------------------------
# Tensor-name parsing helper. Used by some downstream code that infers
# precision tier from tensor name; not used by the planner itself.
# ---------------------------------------------------------------------------
def catalog_idx_for(tensor_id: str) -> int:
    """Parse residual tier index out of a tensor name. Skeleton -> 0."""
    if tensor_id.endswith(".skeleton"):
        return 0
    suffix = tensor_id.rsplit(".", 1)[-1]
    if suffix.startswith("residual_"):
        try:
            return int(suffix[len("residual_"):])
        except ValueError:
            return 0
    return 0
