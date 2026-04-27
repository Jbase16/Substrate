from __future__ import annotations
import logging
import threading
from dataclasses import dataclass
from typing import Mapping, Protocol
from substrate.compiler.ir import EvictRule, ScheduledOp, TensorMetadata

log = logging.getLogger(__name__)

class TensorStore(Protocol):
    def materialize(self, tensor_id: str) -> object: ...
    def release(self, tensor_id: str) -> None: ...

@dataclass
class ResidencyEvent:
    kind: str
    tensor_id: str
    bytes_delta: int
    detail: str = ""

class ResidencyManager:
    def __init__(self, catalog: Mapping[str, TensorMetadata], ram_budget_bytes: int, store: TensorStore):
        self._catalog = catalog
        self._budget = ram_budget_bytes
        self._store = store
        self._resident: set[str] = set()
        self._resident_bytes = 0
        self._pinned: set[str] = set()
        self._lock = threading.RLock()

    def is_resident(self, tensor_id: str) -> bool:
        with self._lock:
            return tensor_id in self._resident

    @property
    def resident_bytes(self) -> int:
        with self._lock:
            return self._resident_bytes

    @property
    def headroom_bytes(self) -> int:
        with self._lock:
            return self._budget - self._resident_bytes

    def pin(self, tensor_id: str) -> ResidencyEvent:
        with self._lock:
            ev = self._ensure_resident_locked(tensor_id, reason="pin")
            self._pinned.add(tensor_id)
            return ev

    def load(self, tensor_id: str) -> ResidencyEvent:
        with self._lock:
            return self._ensure_resident_locked(tensor_id, reason="prefetch")

    def evict(self, tensor_id: str) -> ResidencyEvent:
        with self._lock:
            if tensor_id in self._pinned:
                return ResidencyEvent(kind="rejected", tensor_id=tensor_id, bytes_delta=0, detail="pinned")
            if tensor_id not in self._resident:
                return ResidencyEvent(kind="rejected", tensor_id=tensor_id, bytes_delta=0, detail="not_resident")
            meta = self._catalog[tensor_id]
            self._resident.discard(tensor_id)
            self._resident_bytes -= meta.bytes_in_ram
            self._store.release(tensor_id)
            return ResidencyEvent(kind="evicted", tensor_id=tensor_id, bytes_delta=-meta.bytes_in_ram)

    def apply_evict_rules(self, completed_op: ScheduledOp) -> list[ResidencyEvent]:
        events = []
        for rule in completed_op.evict_after:
            if rule.after_op == completed_op.op_id:
                events.append(self.evict(rule.tensor.tensor_id))
        return events

    def _ensure_resident_locked(self, tensor_id: str, reason: str) -> ResidencyEvent:
        if tensor_id in self._resident:
            return ResidencyEvent(kind="loaded", tensor_id=tensor_id, bytes_delta=0, detail=f"{reason}_already_resident")
        try:
            meta = self._catalog[tensor_id]
        except KeyError:
            raise KeyError(f"Tensor {tensor_id} not in catalog")
        if self._resident_bytes + meta.bytes_in_ram > self._budget:
            raise RuntimeError(
                f"Loading {tensor_id} ({meta.bytes_in_ram / 1e6:.1f} MB) would exceed RAM budget. "
                f"Resident: {self._resident_bytes / 1e9:.2f} GB, budget: {self._budget / 1e9:.2f} GB."
            )
        self._store.materialize(tensor_id)
        self._resident.add(tensor_id)
        self._resident_bytes += meta.bytes_in_ram
        return ResidencyEvent(kind="loaded", tensor_id=tensor_id, bytes_delta=meta.bytes_in_ram, detail=reason)
