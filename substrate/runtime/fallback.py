from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Protocol
from substrate.compiler.ir import FallbackStrategy, ScheduledOp, TensorRef
from substrate.runtime.residency import ResidencyManager

log = logging.getLogger(__name__)

@dataclass
class FallbackDecision:
    op_id: str
    strategy: FallbackStrategy
    missed_tensors: tuple
    effective_requires: tuple
    quality_inflation: float

class SharedExpertProvider(Protocol):
    def shared_expert_for(self, missed_op_id: str, missed_tensor: str) -> TensorRef: ...

class FallbackPlanner:
    def __init__(self, shared_expert: SharedExpertProvider | None = None):
        self._shared_expert = shared_expert

    def decide(self, op: ScheduledOp, missed_tensors: tuple, residency: ResidencyManager) -> FallbackDecision:
        if not missed_tensors:
            return FallbackDecision(op.op_id, op.fallback, (), op.requires, 1.0)

        log.warning("op %s missed %d tensor(s) — applying %s", op.op_id, len(missed_tensors), op.fallback.value)

        if op.fallback is FallbackStrategy.SKELETON_ONLY:
            return self._skeleton_only(op, missed_tensors, residency)
        if op.fallback is FallbackStrategy.SHARED_EXPERT:
            return self._do_shared_expert(op, missed_tensors, residency)
        if op.fallback is FallbackStrategy.STALL:
            return FallbackDecision(op.op_id, FallbackStrategy.STALL, missed_tensors, op.requires, 1.0)
        if op.fallback is FallbackStrategy.ABORT_TOKEN:
            return FallbackDecision(op.op_id, FallbackStrategy.ABORT_TOKEN, missed_tensors, (), float("inf"))
        raise ValueError(f"Unknown fallback strategy: {op.fallback}")

    @staticmethod
    def _skeleton_only(op, missed, residency):
        skeleton_tensors = tuple(t for t in op.requires if t.tensor_id.endswith(".skeleton"))
        if not skeleton_tensors:
            raise RuntimeError(f"op {op.op_id}: SKELETON_ONLY fallback but no skeleton tensor in requires")
        for sk in skeleton_tensors:
            if not residency.is_resident(sk.tensor_id):
                raise RuntimeError(f"op {op.op_id}: skeleton {sk.tensor_id} not resident at fallback time")
        return FallbackDecision(
            op.op_id, FallbackStrategy.SKELETON_ONLY, missed,
            skeleton_tensors, 1.0 + 0.3 * len(missed),
        )

    def _do_shared_expert(self, op, missed, residency):
        if self._shared_expert is None:
            log.error("op %s: SHARED_EXPERT fallback requested but no provider configured; cascading to skeleton", op.op_id)
            return self._skeleton_only(op, missed, residency)
        subs = []
        for tid in missed:
            tref = self._shared_expert.shared_expert_for(op.op_id, tid)
            if not residency.is_resident(tref.tensor_id):
                log.warning("shared expert %s not resident; cascading to skeleton-only", tref.tensor_id)
                return self._skeleton_only(op, missed, residency)
            subs.append(tref)
        skel = tuple(t for t in op.requires if t.tensor_id.endswith(".skeleton"))
        return FallbackDecision(op.op_id, FallbackStrategy.SHARED_EXPERT, missed, skel + tuple(subs), 1.5)
