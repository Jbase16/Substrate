from __future__ import annotations
import logging
from dataclasses import dataclass
from enum import IntEnum
from substrate.compiler.ir import OpBundle, PlanBundle, ScheduledOp

log = logging.getLogger(__name__)

class TierState(IntEnum):
    DEFAULT = 0
    ESCALATED_1 = 1
    ESCALATED_2 = 2

@dataclass
class _OpRuntimeState:
    op_id: str
    current_tier: TierState = TierState.DEFAULT
    persistence_remaining: int = 0
    consecutive_disagreement_hits: int = 0
    last_disagreement: float = 0.0

@dataclass(frozen=True)
class TierTransition:
    op_id: str
    from_tier: int
    to_tier: int
    reason: str
    pool_after_bytes: int

class TierController:
    """
    State machine over compiled tiers. NOT a replanner.
    Constraint inversion: compile all legal responses first -> problem occurs -> switch path.
    """
    def __init__(self, plan: PlanBundle) -> None:
        self.plan = plan
        self.policy = plan.escalation_policy
        self._states = {ob.op_id: _OpRuntimeState(op_id=ob.op_id) for ob in plan.op_bundles}
        self._bundle_by_id = {ob.op_id: ob for ob in plan.op_bundles}
        self._pool_used_bytes = 0
        self._pool_size_bytes = plan.escalation_ram_pool_bytes
        self._transitions: list[TierTransition] = []

    def active_op(self, op_id: str) -> ScheduledOp:
        st = self._states[op_id]
        bundle = self._bundle_by_id[op_id]
        return bundle.at(min(st.current_tier.value, bundle.num_tiers - 1))

    @property
    def pool_used_bytes(self) -> int:
        return self._pool_used_bytes

    @property
    def pool_size_bytes(self) -> int:
        return self._pool_size_bytes

    def transitions_since_reset(self):
        out = tuple(self._transitions)
        self._transitions = []
        return out

    def observe(self, op_id: str, disagreement: float) -> None:
        st = self._states[op_id]
        st.last_disagreement = disagreement
        if disagreement >= self.policy.disagreement_threshold:
            st.consecutive_disagreement_hits += 1
            self._maybe_escalate(op_id)
        else:
            st.consecutive_disagreement_hits = 0

    def end_token(self) -> None:
        if not self.policy.enable_demotion:
            return
        for op_id, st in list(self._states.items()):
            if st.current_tier is TierState.DEFAULT:
                continue
            if st.persistence_remaining > 0:
                st.persistence_remaining -= 1
            if st.persistence_remaining == 0:
                self._step_down(op_id, reason="persistence_expired")

    def force_demote_all(self, reason: str) -> None:
        log.warning("force_demote_all: %s", reason)
        for op_id, st in list(self._states.items()):
            while st.current_tier is not TierState.DEFAULT:
                self._step_down(op_id, reason=f"force:{reason}")

    def _maybe_escalate(self, op_id: str) -> None:
        st = self._states[op_id]
        bundle = self._bundle_by_id[op_id]
        target = TierState.ESCALATED_1
        if (st.consecutive_disagreement_hits >= self.policy.consecutive_hits_for_tier_2
                and bundle.num_tiers >= 3):
            target = TierState.ESCALATED_2
        if st.current_tier.value >= target.value:
            st.persistence_remaining = self.policy.persistence_tokens
            return
        if target.value >= bundle.num_tiers:
            target = TierState(bundle.num_tiers - 1)
            if target.value <= st.current_tier.value:
                return
        while st.current_tier.value < target.value:
            next_tier_idx = st.current_tier.value + 1
            if next_tier_idx >= bundle.num_tiers:
                break
            cur_op = bundle.at(st.current_tier.value)
            next_op = bundle.at(next_tier_idx)
            delta = next_op.peak_ram_delta_bytes - cur_op.peak_ram_delta_bytes
            if self._pool_used_bytes + delta > self._pool_size_bytes:
                if not self._evict_for_headroom(delta, exclude=op_id):
                    log.debug("op %s escalation refused: pool full", op_id)
                    return
            self._pool_used_bytes += delta
            from_tier = st.current_tier.value
            st.current_tier = TierState(next_tier_idx)
            st.persistence_remaining = self.policy.persistence_tokens
            self._transitions.append(TierTransition(
                op_id=op_id, from_tier=from_tier, to_tier=next_tier_idx,
                reason="disagreement", pool_after_bytes=self._pool_used_bytes,
            ))

    def _step_down(self, op_id: str, reason: str) -> None:
        st = self._states[op_id]
        bundle = self._bundle_by_id[op_id]
        if st.current_tier is TierState.DEFAULT:
            return
        cur_op = bundle.at(st.current_tier.value)
        target_idx = st.current_tier.value - 1
        next_op = bundle.at(target_idx)
        delta = cur_op.peak_ram_delta_bytes - next_op.peak_ram_delta_bytes
        self._pool_used_bytes = max(0, self._pool_used_bytes - delta)
        from_tier = st.current_tier.value
        st.current_tier = TierState(target_idx)
        st.persistence_remaining = self.policy.persistence_tokens if target_idx > 0 else 0
        self._transitions.append(TierTransition(
            op_id=op_id, from_tier=from_tier, to_tier=target_idx,
            reason=reason, pool_after_bytes=self._pool_used_bytes,
        ))

    def _evict_for_headroom(self, needed: int, exclude: str) -> bool:
        candidates = [
            st for op_id, st in self._states.items()
            if op_id != exclude
            and st.current_tier is not TierState.DEFAULT
            and st.last_disagreement < self.policy.disagreement_threshold * 0.5
        ]
        candidates.sort(key=lambda s: (s.last_disagreement, -s.persistence_remaining))
        freed = 0
        for st in candidates:
            bundle = self._bundle_by_id[st.op_id]
            freed_by_demote = bundle.at(st.current_tier.value).peak_ram_delta_bytes
            while st.current_tier is not TierState.DEFAULT:
                self._step_down(st.op_id, reason="evicted_for_headroom")
            freed += freed_by_demote
            if freed >= needed:
                return True
        return False
