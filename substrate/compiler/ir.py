"""
substrate.compiler.ir — Plan IR with the escalation ladder as a first-class structure.

The compiler does not emit a single execution plan. It emits a PlanBundle:
a default schedule plus a set of pre-validated escalation tiers per op.
Runtime adaptation is then a state-machine over compiled tiers, not an
optimization solve.

Hierarchy:

    PlanBundle
        ├─ budget                       # The hard envelope
        ├─ tensor_catalog               # All tensors, all tiers, sizes & locations
        ├─ op_bundles[]                 # Per-op tier ladder
        │      ├─ tiers[0] (default)    # Cheapest valid; the path runs by default
        │      ├─ tiers[1] (escalation) # Better quality, more RAM/bandwidth
        │      └─ tiers[2] (ceiling)    # Best within the per-op pool budget
        ├─ escalation_policy            # When to step up
        ├─ fallback_policy              # When to give up on tiers and degrade
        ├─ escalation_ram_pool_bytes    # Total RAM available for active escalations
        └─ predicted_*                  # Aggregate metrics for the default path

Each ScheduledOp carries `tier_index` so the executor and trace recorder
know which tier it represents. tiers[0].tier_index == 0; tiers[1].tier_index == 1.

Validation invariants (enforced in __post_init__):

    1. Every tensor reference resolves.
    2. Topological ordering: every prefetch's start_during precedes its deadline_before.
    3. The default schedule (tiers[0] for every op) fits the RAM budget.
    4. The escalation pool is non-negative.
    5. Each tier's quality_risk is monotonically non-increasing in tier_index
       (escalating must not make quality worse).
    6. Each tier's peak_ram_delta_bytes is monotonically non-decreasing in
       tier_index (escalating costs at least as much RAM as the previous tier).

The compiler is responsible for ensuring that for any combination of K
op escalations (K bounded by escalation_policy.max_concurrent_escalations),
the total RAM stays under budget. The IR validates the *static* defaults;
the *dynamic* envelope is enforced by the tier_controller using the
escalation_ram_pool_bytes field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class OpKind(str, Enum):
    EMBEDDING = "embedding"
    NORM = "norm"
    ATTENTION = "attention"
    MLP_DENSE = "mlp_dense"
    MOE_ROUTER = "moe_router"
    MOE_EXPERT = "moe_expert"
    MOE_DISPATCH = "moe_dispatch"
    LM_HEAD = "lm_head"


class FallbackStrategy(str, Enum):
    SKELETON_ONLY = "skeleton_only"
    SHARED_EXPERT = "shared_expert"
    ABORT_TOKEN = "abort_token"
    STALL = "stall"


@dataclass(frozen=True)
class TensorRef:
    tensor_id: str

    def __post_init__(self) -> None:
        if not self.tensor_id or any(c.isspace() for c in self.tensor_id):
            raise ValueError(f"Invalid tensor_id: {self.tensor_id!r}")


@dataclass(frozen=True)
class TensorMetadata:
    tensor_id: str
    bytes_in_ram: int
    bytes_on_ssd: int
    is_skeleton: bool
    layer_id: int
    tier_index: int

    def __post_init__(self) -> None:
        if self.bytes_in_ram < 0 or self.bytes_on_ssd < 0:
            raise ValueError(f"{self.tensor_id}: byte counts must be >= 0")
        if self.is_skeleton and self.tier_index != 0:
            raise ValueError(f"{self.tensor_id}: skeleton must have tier_index=0")
        if not self.is_skeleton and self.bytes_on_ssd == 0:
            raise ValueError(
                f"{self.tensor_id}: non-skeleton tensor must have bytes_on_ssd > 0"
            )


@dataclass(frozen=True)
class PrefetchRequest:
    tensor: TensorRef
    start_during: str
    deadline_before: str
    priority: int = 0

    def __post_init__(self) -> None:
        if self.start_during == self.deadline_before:
            raise ValueError(
                f"PrefetchRequest({self.tensor.tensor_id}): start_during must "
                f"differ from deadline_before"
            )


@dataclass(frozen=True)
class EvictRule:
    tensor: TensorRef
    after_op: str


@dataclass(frozen=True)
class ScheduledOp:
    op_id: str
    tier_index: int
    op_kind: OpKind
    layer_id: int
    requires: tuple[TensorRef, ...]
    prefetch: tuple[PrefetchRequest, ...]
    evict_after: tuple[EvictRule, ...]
    fallback: FallbackStrategy
    estimated_compute_us: int
    estimated_quality_risk: float
    peak_ram_delta_bytes: int = 0
    moe_likely_experts: tuple[str, ...] = ()
    moe_top_k: int = 0

    def __post_init__(self) -> None:
        if not self.op_id:
            raise ValueError("op_id must be non-empty")
        if self.tier_index < 0:
            raise ValueError(f"{self.op_id}: tier_index must be >= 0")
        if not (0.0 <= self.estimated_quality_risk <= 1.0):
            raise ValueError(
                f"{self.op_id}@tier{self.tier_index}: estimated_quality_risk "
                f"out of [0,1]: {self.estimated_quality_risk}"
            )
        if self.estimated_compute_us < 0:
            raise ValueError(f"{self.op_id}: estimated_compute_us must be >= 0")
        if self.peak_ram_delta_bytes < 0:
            raise ValueError(f"{self.op_id}: peak_ram_delta_bytes must be >= 0")
        if self.tier_index == 0 and self.peak_ram_delta_bytes != 0:
            raise ValueError(
                f"{self.op_id}: tier 0 must have peak_ram_delta_bytes == 0"
            )
        if self.op_kind is OpKind.MOE_DISPATCH:
            if self.moe_top_k <= 0:
                raise ValueError(f"{self.op_id}: MOE_DISPATCH requires moe_top_k > 0")
        else:
            if self.moe_top_k != 0 or self.moe_likely_experts:
                raise ValueError(
                    f"{self.op_id}: moe_* fields only valid on MOE_DISPATCH"
                )


@dataclass(frozen=True)
class OpBundle:
    """All tiers for one logical op, ordered ascending in cost/quality."""
    op_id: str
    tiers: tuple[ScheduledOp, ...]

    def __post_init__(self) -> None:
        if len(self.tiers) == 0:
            raise ValueError(f"{self.op_id}: bundle must have at least one tier")
        for i, t in enumerate(self.tiers):
            if t.op_id != self.op_id:
                raise ValueError(
                    f"OpBundle {self.op_id} tier {i} has mismatched op_id={t.op_id}"
                )
            if t.tier_index != i:
                raise ValueError(
                    f"OpBundle {self.op_id}: tiers[{i}].tier_index must == {i}, "
                    f"got {t.tier_index}"
                )
        risks = [t.estimated_quality_risk for t in self.tiers]
        if any(risks[i] < risks[i + 1] for i in range(len(risks) - 1)):
            raise ValueError(
                f"OpBundle {self.op_id}: quality_risk must be monotonically "
                f"non-increasing in tier_index, got {risks}"
            )
        deltas = [t.peak_ram_delta_bytes for t in self.tiers]
        if any(deltas[i] > deltas[i + 1] for i in range(len(deltas) - 1)):
            raise ValueError(
                f"OpBundle {self.op_id}: peak_ram_delta_bytes must be "
                f"monotonically non-decreasing, got {deltas}"
            )

    @property
    def default(self) -> ScheduledOp:
        return self.tiers[0]

    def at(self, tier_index: int) -> ScheduledOp:
        if not (0 <= tier_index < len(self.tiers)):
            raise IndexError(
                f"OpBundle {self.op_id}: tier_index {tier_index} out of range "
                f"(have {len(self.tiers)} tiers)"
            )
        return self.tiers[tier_index]

    @property
    def num_tiers(self) -> int:
        return len(self.tiers)


@dataclass(frozen=True)
class EscalationPolicy:
    """When the tier_controller decides to escalate / demote ops."""
    disagreement_threshold: float = 0.15
    consecutive_hits_for_tier_2: int = 3
    persistence_tokens: int = 32
    enable_demotion: bool = True
    max_concurrent_escalations: int = 8

    def __post_init__(self) -> None:
        if self.disagreement_threshold < 0:
            raise ValueError("disagreement_threshold must be >= 0")
        if self.persistence_tokens <= 0:
            raise ValueError("persistence_tokens must be > 0")
        if self.max_concurrent_escalations < 0:
            raise ValueError("max_concurrent_escalations must be >= 0")


@dataclass(frozen=True)
class FallbackPolicy:
    """What happens when even the best tier fails."""
    deadline_miss_strategy: FallbackStrategy = FallbackStrategy.SKELETON_ONLY
    critical_latency_factor: float = 2.0
    critical_ssd_bw_factor: float = 1.5

    def __post_init__(self) -> None:
        if self.critical_latency_factor < 1.0:
            raise ValueError("critical_latency_factor must be >= 1.0")
        if self.critical_ssd_bw_factor < 1.0:
            raise ValueError("critical_ssd_bw_factor must be >= 1.0")


@dataclass(frozen=True)
class Budget:
    max_ram_bytes: int
    max_ssd_cache_bytes: int
    sustained_ssd_bw_bytes_per_sec: int
    quality_loss_cap: float
    target_tokens_per_second: float | None = None

    def __post_init__(self) -> None:
        if self.max_ram_bytes <= 0 or self.max_ssd_cache_bytes <= 0:
            raise ValueError("Budget byte fields must be > 0")
        if self.sustained_ssd_bw_bytes_per_sec <= 0:
            raise ValueError("sustained_ssd_bw_bytes_per_sec must be > 0")
        if not (0.0 <= self.quality_loss_cap <= 1.0):
            raise ValueError("quality_loss_cap must be in [0,1]")


@dataclass(frozen=True)
class PlanBundle:
    model_id: str
    budget: Budget
    tensor_catalog: Mapping[str, TensorMetadata]
    op_bundles: tuple[OpBundle, ...]
    escalation_policy: EscalationPolicy
    fallback_policy: FallbackPolicy
    predicted_peak_resident_bytes: int
    predicted_steady_state_resident_bytes: int
    predicted_ssd_bandwidth_bps: int
    predicted_tokens_per_second: float
    predicted_quality_loss: float
    escalation_ram_pool_bytes: int
    solver_version: str
    solver_notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.op_bundles:
            raise ValueError("PlanBundle must contain at least one op_bundle")

        op_pos: dict[str, int] = {}
        for i, ob in enumerate(self.op_bundles):
            if ob.op_id in op_pos:
                raise ValueError(f"Duplicate op_id in bundle: {ob.op_id}")
            op_pos[ob.op_id] = i

        catalog_ids = set(self.tensor_catalog.keys())

        for ob in self.op_bundles:
            for tier_op in ob.tiers:
                for tref in tier_op.requires:
                    if tref.tensor_id not in catalog_ids:
                        raise ValueError(
                            f"{tier_op.op_id}@tier{tier_op.tier_index}: requires "
                            f"unknown tensor {tref.tensor_id}"
                        )
                for pf in tier_op.prefetch:
                    if pf.tensor.tensor_id not in catalog_ids:
                        raise ValueError(
                            f"{tier_op.op_id}@tier{tier_op.tier_index}: prefetch "
                            f"references unknown tensor {pf.tensor.tensor_id}"
                        )
                    if pf.start_during not in op_pos or pf.deadline_before not in op_pos:
                        raise ValueError(
                            f"{tier_op.op_id}@tier{tier_op.tier_index}: prefetch "
                            f"references unknown op_id"
                        )
                    if op_pos[pf.start_during] >= op_pos[pf.deadline_before]:
                        raise ValueError(
                            f"{tier_op.op_id}@tier{tier_op.tier_index}: "
                            f"prefetch start_during={pf.start_during} must "
                            f"precede deadline_before={pf.deadline_before}"
                        )
                for ev in tier_op.evict_after:
                    if ev.tensor.tensor_id not in catalog_ids:
                        raise ValueError(
                            f"{tier_op.op_id}@tier{tier_op.tier_index}: evict_after "
                            f"references unknown tensor {ev.tensor.tensor_id}"
                        )
                    if ev.after_op not in op_pos:
                        raise ValueError(
                            f"{tier_op.op_id}@tier{tier_op.tier_index}: evict_after "
                            f"references unknown op_id {ev.after_op}"
                        )

        if self.predicted_peak_resident_bytes > self.budget.max_ram_bytes:
            raise ValueError(
                f"PlanBundle invalid: default path peak resident "
                f"{self.predicted_peak_resident_bytes} > budget "
                f"{self.budget.max_ram_bytes}"
            )

        if self.escalation_ram_pool_bytes < 0:
            raise ValueError("escalation_ram_pool_bytes must be >= 0")
        max_pool = self.budget.max_ram_bytes - self.predicted_peak_resident_bytes
        if self.escalation_ram_pool_bytes > max_pool:
            raise ValueError(
                f"escalation_ram_pool_bytes={self.escalation_ram_pool_bytes} "
                f"exceeds available headroom {max_pool}"
            )

        K = self.escalation_policy.max_concurrent_escalations
        if K > 0:
            tier1_deltas = sorted(
                (ob.tiers[1].peak_ram_delta_bytes
                 for ob in self.op_bundles if ob.num_tiers >= 2),
                reverse=True,
            )
            top_k_sum = sum(tier1_deltas[:K])
            if top_k_sum > self.escalation_ram_pool_bytes:
                raise ValueError(
                    f"PlanBundle invalid: K={K} largest tier-1 escalations "
                    f"need {top_k_sum} bytes, pool has only "
                    f"{self.escalation_ram_pool_bytes}. The compiler must "
                    f"either shrink tier-1 deltas, raise the pool, or lower K."
                )

        if self.predicted_quality_loss > self.budget.quality_loss_cap:
            raise ValueError(
                f"PlanBundle invalid: predicted_quality_loss "
                f"{self.predicted_quality_loss} > cap "
                f"{self.budget.quality_loss_cap}"
            )

    def bundle(self, op_id: str) -> OpBundle:
        for ob in self.op_bundles:
            if ob.op_id == op_id:
                return ob
        raise KeyError(f"OpBundle not found: {op_id}")

    def tensor(self, tensor_id: str) -> TensorMetadata:
        try:
            return self.tensor_catalog[tensor_id]
        except KeyError:
            raise KeyError(f"Tensor not in catalog: {tensor_id}")

    @property
    def default_schedule(self) -> tuple[ScheduledOp, ...]:
        return tuple(ob.default for ob in self.op_bundles)

    @property
    def num_ops(self) -> int:
        return len(self.op_bundles)
