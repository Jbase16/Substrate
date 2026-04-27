from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from enum import Enum
from substrate.compiler.ir import PlanBundle

class DriftKind(str, Enum):
    LATENCY_WARN = "latency_warn"
    LATENCY_CRITICAL = "latency_critical"
    SSD_BW_CRITICAL = "ssd_bw_critical"

@dataclass(frozen=True)
class DriftSignal:
    kind: DriftKind
    magnitude: float
    detail: str

class RuntimeMonitor:
    def __init__(self, plan: PlanBundle, latency_warn_factor: float = 1.3,
                 latency_window: int = 64, bw_window: int = 32):
        self.plan = plan
        self._latency_warn = latency_warn_factor
        self._latency_critical = plan.fallback_policy.critical_latency_factor
        self._ssd_critical = plan.fallback_policy.critical_ssd_bw_factor
        self._latency_window: deque[int] = deque(maxlen=latency_window)
        self._bw_window: deque[int] = deque(maxlen=bw_window)
        self.tokens_seen = 0

    def record_op(self, predicted_us: int, measured_us: int) -> None:
        if predicted_us > 0:
            self._latency_window.append(measured_us * 1000 // predicted_us)

    def record_ssd_bw_bps(self, measured_bps: int) -> None:
        self._bw_window.append(measured_bps)

    def end_token(self) -> None:
        self.tokens_seen += 1

    def assess(self):
        if len(self._latency_window) < min(16, self._latency_window.maxlen or 16):
            return None
        avg_ratio = sum(self._latency_window) / (len(self._latency_window) * 1000)
        if avg_ratio >= self._latency_critical:
            return DriftSignal(DriftKind.LATENCY_CRITICAL, avg_ratio, f"latency_avg={avg_ratio:.2f}x predicted")
        if self._bw_window:
            avg_bw = sum(self._bw_window) / len(self._bw_window)
            cap = self.plan.budget.sustained_ssd_bw_bytes_per_sec
            if avg_bw > cap * self._ssd_critical:
                return DriftSignal(DriftKind.SSD_BW_CRITICAL, avg_bw / cap, f"ssd_bw={avg_bw / 1e9:.2f}GB/s")
        if avg_ratio >= self._latency_warn:
            return DriftSignal(DriftKind.LATENCY_WARN, avg_ratio, f"latency_avg={avg_ratio:.2f}x predicted")
        return None
