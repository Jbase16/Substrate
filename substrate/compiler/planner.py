from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from substrate.compiler.feasibility import (
    FeasibilityReport, InfeasibleBudgetError, check_feasibility,
)
from substrate.compiler.ir import (
    Budget, EscalationPolicy, EvictRule, FallbackPolicy, FallbackStrategy,
    OpBundle, OpKind, PlanBundle, PrefetchRequest, ScheduledOp,
    TensorMetadata, TensorRef,
)
from substrate.compiler.quality import QualityEstimator

log = logging.getLogger(__name__)
SOLVER_VERSION = "0.1.0"


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
    Precision.SKELETON_2BIT, Precision.SKELETON_PLUS_R1,
    Precision.SKELETON_PLUS_R2, Precision.REFINED_6BIT, Precision.NEAR_FP16,
)


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


@dataclass
class PlannerOptions:
    max_upgrade_iterations: int = 2000
    moe_prefetch_top_k: int = 4
    emit_escalation_tiers: bool = True


class Planner:
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

    def compile(self, profile: ModelProfile, budget: Budget) -> PlanBundle:
        notes: list[str] = []
        report = check_feasibility(profile, budget)
        if not report.feasible:
            raise InfeasibleBudgetError(report)
        notes.append(f"feasibility ok: floor_ram={report.floor_resident_bytes / 1e9:.2f}GB")

        tier0 = self._greedy_quality_fill(profile, budget, notes)

        if self.opts.emit_escalation_tiers:
            tier1, tier2 = self._build_escalation_precisions(profile, tier0)
        else:
            tier1, tier2 = {}, {}

        catalog = self._build_catalog(profile, tier0, tier1, tier2)
        residency = self._assign_residency(profile, budget, catalog, tier0, notes)
        steady_ram, peak_ram = self._compute_ram_envelope(profile, catalog, residency)
        pool = max(0, budget.max_ram_bytes - peak_ram)
        notes.append(
            f"escalation_pool: {pool / 1e6:.1f}MB "
            f"(peak_resident={peak_ram / 1e9:.2f}GB, max_ram={budget.max_ram_bytes / 1e9:.2f}GB)"
        )
        admitted_tier1, admitted_tier2 = self._admit_escalations_to_pool(
            profile, tier0, tier1, tier2, pool, notes
        )
        op_bundles = self._build_bundles(profile, catalog, residency, tier0, admitted_tier1, admitted_tier2)
        ssd_bw = self._compute_ssd_bandwidth(profile, catalog, residency, budget)
        compute_us = sum(ob.default.estimated_compute_us for ob in op_bundles)
        tps = 1e6 / max(1, compute_us)
        quality_loss = min(1.0, sum(ob.default.estimated_quality_risk for ob in op_bundles))

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

    def _greedy_quality_fill(self, profile, budget, notes):
        precision = {op.op_id: Precision.SKELETON_2BIT for op in profile.ops}
        op_by_id = {op.op_id: op for op in profile.ops}

        def loss(op_id, p):
            return 0.005 / p.effective_bits * (1.0 + op_by_id[op_id].sensitivity)

        def ram(op_id, p):
            return op_by_id[op_id].bytes_at_precision(p)

        def total_loss():
            return sum(loss(oid, p) for oid, p in precision.items())

        def total_ram():
            base = profile.embedding_bytes + profile.lm_head_bytes + profile.runtime_overhead_bytes
            return base + sum(ram(oid, p) for oid, p in precision.items())

        for it in range(self.opts.max_upgrade_iterations):
            if total_loss() <= budget.quality_loss_cap * 0.95:
                notes.append(f"tier-0 fill converged at iter {it}")
                return precision
            best_op_id = best_target = None
            best_score = 0.0
            current_ram = total_ram()
            for op_id, cur_p in precision.items():
                idx = _PRECISION_LADDER.index(cur_p)
                if idx + 1 >= len(_PRECISION_LADDER):
                    continue
                nxt = _PRECISION_LADDER[idx + 1]
                d_ram = ram(op_id, nxt) - ram(op_id, cur_p)
                if d_ram <= 0 or current_ram + d_ram > budget.max_ram_bytes:
                    continue
                d_loss = loss(op_id, cur_p) - loss(op_id, nxt)
                if d_loss <= 0:
                    continue
                score = (d_loss / d_ram) * (1.0 + op_by_id[op_id].sensitivity)
                if score > best_score:
                    best_score, best_op_id, best_target = score, op_id, nxt
            if best_op_id is None:
                notes.append(f"tier-0 fill stuck at iter {it} loss={total_loss():.4f}")
                return precision
            precision[best_op_id] = best_target
        notes.append("tier-0 fill: max_upgrade_iterations exhausted")
        return precision

    @staticmethod
    def _build_escalation_precisions(profile, tier0):
        tier1, tier2 = {}, {}
        for op in profile.ops:
            idx = _PRECISION_LADDER.index(tier0[op.op_id])
            if idx + 1 < len(_PRECISION_LADDER):
                tier1[op.op_id] = _PRECISION_LADDER[idx + 1]
            if idx + 2 < len(_PRECISION_LADDER):
                tier2[op.op_id] = _PRECISION_LADDER[idx + 2]
        return tier1, tier2

    @staticmethod
    def _build_catalog(profile, tier0, tier1, tier2):
        catalog = {}
        for op in profile.ops:
            highest = tier0[op.op_id]
            for p in (tier1.get(op.op_id), tier2.get(op.op_id)):
                if p and _PRECISION_LADDER.index(p) > _PRECISION_LADDER.index(highest):
                    highest = p
            highest_idx = _PRECISION_LADDER.index(highest)

            sk_id = f"{op.op_id}.skeleton"
            sk_bytes = op.bytes_at_precision(Precision.SKELETON_2BIT)
            catalog[sk_id] = TensorMetadata(
                tensor_id=sk_id, bytes_in_ram=sk_bytes, bytes_on_ssd=sk_bytes,
                is_skeleton=True, layer_id=op.layer_id, tier_index=0,
            )
            for t in range(1, highest_idx + 1):
                tier_prec = _PRECISION_LADDER[t]
                residual_bytes = max(0, op.bytes_at_precision(tier_prec) - op.bytes_at_precision(_PRECISION_LADDER[t-1]))
                if residual_bytes == 0:
                    continue
                rid = f"{op.op_id}.residual_{t}"
                catalog[rid] = TensorMetadata(
                    tensor_id=rid, bytes_in_ram=residual_bytes, bytes_on_ssd=residual_bytes,
                    is_skeleton=False, layer_id=op.layer_id, tier_index=t,
                )
        return catalog

    @staticmethod
    def _assign_residency(profile, budget, catalog, tier0, notes):
        op_by_id = {op.op_id: op for op in profile.ops}
        residency = {}
        ram_used = profile.embedding_bytes + profile.lm_head_bytes + profile.runtime_overhead_bytes

        for tid, meta in catalog.items():
            if meta.is_skeleton:
                residency[tid] = "resident"
                ram_used += meta.bytes_in_ram

        tier0_required, higher_tier = [], []
        for tid, meta in catalog.items():
            if meta.is_skeleton:
                continue
            owner = tid.rsplit(".", 1)[0]
            if meta.tier_index <= _PRECISION_LADDER.index(tier0[owner]):
                tier0_required.append((tid, meta))
            else:
                higher_tier.append((tid, meta))

        tier0_required.sort(key=lambda x: (-op_by_id.get(x[0].rsplit(".",1)[0], type("", (), {"sensitivity": 0})).sensitivity if x[0].rsplit(".",1)[0] in op_by_id else 0, x[1].tier_index))

        rc = sc = 0
        for tid, meta in tier0_required:
            if ram_used + meta.bytes_in_ram <= budget.max_ram_bytes:
                residency[tid] = "resident"
                ram_used += meta.bytes_in_ram
                rc += 1
            else:
                residency[tid] = "streamed"
                sc += 1
        for tid, _ in higher_tier:
            residency[tid] = "streamed"

        notes.append(f"residency: {rc} tier-0 residuals resident, {sc} streamed, {len(higher_tier)} higher-tier streamed")
        return residency

    def _admit_escalations_to_pool(self, profile, tier0, tier1, tier2, pool, notes):
        op_by_id = {op.op_id: op for op in profile.ops}
        K = self.escalation_policy.max_concurrent_escalations

        def delta(op_id, target):
            return op_by_id[op_id].bytes_at_precision(target) - op_by_id[op_id].bytes_at_precision(tier0[op_id])

        candidates = sorted(tier1.keys(), key=lambda oid: (-op_by_id[oid].sensitivity, delta(oid, tier1[oid])))
        admitted_tier1, admitted_deltas = {}, []

        def top_k_sum(d):
            return sum(d[:K]) if K > 0 else 0

        for op_id in candidates:
            d = delta(op_id, tier1[op_id])
            if d <= 0:
                admitted_tier1[op_id] = tier1[op_id]
                continue
            new_d = sorted(admitted_deltas + [d], reverse=True)
            if top_k_sum(new_d) <= pool:
                admitted_tier1[op_id] = tier1[op_id]
                admitted_deltas = new_d

        notes.append(f"tier-1 admission: {len(admitted_tier1)}/{len(tier1)} ops; top-{K} delta sum = {top_k_sum(admitted_deltas) / 1e6:.1f}MB / pool {pool / 1e6:.1f}MB")

        admitted_tier2, admitted_t2_deltas = {}, []
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
            new_d = sorted(admitted_t2_deltas + [d], reverse=True)
            if top_k_sum(new_d) <= pool:
                admitted_tier2[op_id] = t2
                admitted_t2_deltas = new_d

        notes.append(f"tier-2 admission: {len(admitted_tier2)}/{len(tier2)} ops")
        return admitted_tier1, admitted_tier2

    def _build_bundles(self, profile, catalog, residency, tier0, tier1, tier2):
        op_list = list(profile.ops)
        op_pos = {op.op_id: i for i, op in enumerate(op_list)}
        bundles = []
        for op in op_list:
            tiers_list = [self._make_op_at_tier(op, op_list, op_pos, catalog, residency, tier0[op.op_id], 0, tier0[op.op_id])]
            if op.op_id in tier1:
                tiers_list.append(self._make_op_at_tier(op, op_list, op_pos, catalog, residency, tier1[op.op_id], 1, tier0[op.op_id]))
            if op.op_id in tier2:
                tiers_list.append(self._make_op_at_tier(op, op_list, op_pos, catalog, residency, tier2[op.op_id], 2, tier0[op.op_id]))
            bundles.append(OpBundle(op_id=op.op_id, tiers=tuple(tiers_list)))
        return bundles

    def _make_op_at_tier(self, op, op_list, op_pos, catalog, residency, precision, tier_index, base_precision):
        prec_idx = _PRECISION_LADDER.index(precision)
        required = [TensorRef(f"{op.op_id}.skeleton")]
        for t in range(1, prec_idx + 1):
            rid = f"{op.op_id}.residual_{t}"
            if rid in catalog:
                required.append(TensorRef(rid))

        prefetches = []
        my_pos = op_pos[op.op_id]
        prev_id = op_list[my_pos - 1].op_id if my_pos > 0 else None
        for tref in required:
            if residency[tref.tensor_id] == "streamed" and prev_id is not None:
                prefetches.append(PrefetchRequest(
                    tensor=tref, start_during=prev_id, deadline_before=op.op_id,
                    priority=1 + catalog[tref.tensor_id].tier_index,
                ))

        evicts = [
            EvictRule(tensor=tref, after_op=op.op_id)
            for tref in required if residency[tref.tensor_id] == "streamed"
        ]

        fallback = (
            FallbackStrategy.SHARED_EXPERT
            if op.op_kind in (OpKind.MOE_EXPERT, OpKind.MOE_DISPATCH)
            else FallbackStrategy.SKELETON_ONLY
        )

        t_frac = prec_idx / (len(_PRECISION_LADDER) - 1)
        est_us = max(1, int(op.skeleton_compute_us + t_frac * (op.full_precision_compute_us - op.skeleton_compute_us)))

        moe_likely, moe_top_k = (), 0
        if op.op_kind is OpKind.MOE_DISPATCH:
            moe_top_k = op.moe_top_k
            k = min(self.opts.moe_prefetch_top_k, op.moe_num_experts)
            moe_likely = tuple(f"{op.op_id}.expert_{j}" for j in range(k))

        risk = min(1.0, 0.005 / precision.effective_bits * (1.0 + op.sensitivity))
        ram_delta = 0 if tier_index == 0 else max(0, op.bytes_at_precision(precision) - op.bytes_at_precision(base_precision))

        return ScheduledOp(
            op_id=op.op_id, tier_index=tier_index, op_kind=op.op_kind,
            layer_id=op.layer_id, requires=tuple(required), prefetch=tuple(prefetches),
            evict_after=tuple(evicts), fallback=fallback, estimated_compute_us=est_us,
            estimated_quality_risk=risk, peak_ram_delta_bytes=ram_delta,
            moe_likely_experts=moe_likely, moe_top_k=moe_top_k,
        )

    @staticmethod
    def _compute_ram_envelope(profile, catalog, residency):
        steady = (
            profile.embedding_bytes + profile.lm_head_bytes + profile.runtime_overhead_bytes
            + sum(meta.bytes_in_ram for tid, meta in catalog.items() if residency[tid] == "resident")
        )
        streamed = [meta.bytes_in_ram for tid, meta in catalog.items() if residency[tid] == "streamed"]
        return steady, steady + (max(streamed) if streamed else 0)

    @staticmethod
    def _compute_ssd_bandwidth(profile, catalog, residency, budget):
        per_token = sum(meta.bytes_on_ssd for tid, meta in catalog.items() if residency[tid] == "streamed")
        return int(per_token * (budget.target_tokens_per_second or 1.0))


def catalog_idx_for(tensor_id: str) -> int:
    if tensor_id.endswith(".skeleton"):
        return 0
    suffix = tensor_id.rsplit(".", 1)[-1]
    if suffix.startswith("residual_"):
        try:
            return int(suffix[len("residual_"):])
        except ValueError:
            return 0
    return 0
