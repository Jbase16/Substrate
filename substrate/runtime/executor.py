from __future__ import annotations
import logging, time
from dataclasses import dataclass
from typing import Protocol
from substrate.compiler.ir import FallbackStrategy, PlanBundle, ScheduledOp
from substrate.runtime.fallback import FallbackDecision, FallbackPlanner
from substrate.runtime.monitor import DriftKind, RuntimeMonitor
from substrate.runtime.prefetch import PrefetchScheduler
from substrate.runtime.residency import ResidencyManager
from substrate.runtime.tier_controller import TierController
from substrate.runtime.verifier import VerifierProbe

log = logging.getLogger(__name__)

class OpKernel(Protocol):
    def execute(self, op: ScheduledOp, decision: FallbackDecision, hidden: object) -> object: ...

class TraceSink(Protocol):
    def record_op(self, op_id, tier, measured_us, predicted_us, fallback) -> None: ...
    def record_token(self, outcome) -> None: ...
    def record_tier_transition(self, op_id, from_tier, to_tier, reason) -> None: ...

@dataclass
class TokenOutcome:
    token_index: int
    total_us: int
    fallback_count: int
    tier_transitions: int
    missed_tensors: tuple
    pool_used_after_token: int

class Executor:
    def __init__(self, plan, kernel, residency, prefetcher, tier_controller, monitor,
                 fallback_planner, verifier=None, trace=None):
        self.plan = plan
        self.kernel = kernel
        self.residency = residency
        self.prefetcher = prefetcher
        self.tier_controller = tier_controller
        self.monitor = monitor
        self.fallback_planner = fallback_planner
        self.verifier = verifier
        self.trace = trace
        self._token_index = 0
        self._cold_start_verified = False

    def start(self):
        self._verify_cold_start()
        self.prefetcher.start()

    def stop(self):
        self.prefetcher.stop()

    def step_token(self, hidden: object) -> object:
        if not self._cold_start_verified:
            self._verify_cold_start()

        token_start = time.perf_counter_ns()
        fallback_count = 0
        missed_total = []

        for ob in self.plan.op_bundles:
            op = self.tier_controller.active_op(ob.op_id)
            self.prefetcher.on_op_start(op)
            required_ids = tuple(t.tensor_id for t in op.requires)
            wait_budget_us = max(100, op.estimated_compute_us)
            missed = self.prefetcher.wait_until_resident(required_ids, wait_budget_us)
            decision = self.fallback_planner.decide(op, missed, self.residency)
            if missed:
                fallback_count += 1
                missed_total.extend(missed)
            if decision.strategy is FallbackStrategy.ABORT_TOKEN:
                self._record_token(token_start, fallback_count, missed_total)
                return None

            t0 = time.perf_counter_ns()
            hidden = self.kernel.execute(op, decision, hidden)
            elapsed_us = (time.perf_counter_ns() - t0) // 1000

            self.monitor.record_op(op.estimated_compute_us, elapsed_us)
            if self.trace is not None:
                self.trace.record_op(op.op_id, op.tier_index, elapsed_us, op.estimated_compute_us,
                                     decision.strategy if missed else None)
            if self.verifier is not None:
                self.tier_controller.observe(op.op_id, self.verifier.disagreement(op.op_id, hidden))

            self.residency.apply_evict_rules(op)
            self.prefetcher.on_op_end(op)

        self.monitor.record_ssd_bw_bps(0)
        self.monitor.end_token()
        self.tier_controller.end_token()

        signal = self.monitor.assess()
        if signal is not None and signal.kind in (DriftKind.LATENCY_CRITICAL, DriftKind.SSD_BW_CRITICAL):
            self.tier_controller.force_demote_all(signal.detail)

        for tt in self.tier_controller.transitions_since_reset():
            if self.trace is not None:
                self.trace.record_tier_transition(tt.op_id, tt.from_tier, tt.to_tier, tt.reason)

        self._record_token(token_start, fallback_count, missed_total)
        return hidden

    def _verify_cold_start(self):
        first_default = self.plan.op_bundles[0].default
        for tref in first_default.requires:
            if not self.residency.is_resident(tref.tensor_id):
                raise RuntimeError(
                    f"Cold start: {tref.tensor_id} required by first op {first_default.op_id} is not resident"
                )
        self._cold_start_verified = True

    def _record_token(self, token_start_ns, fallback_count, missed):
        if self.trace is None:
            self._token_index += 1
            return
        total_us = (time.perf_counter_ns() - token_start_ns) // 1000
        self.trace.record_token(TokenOutcome(
            token_index=self._token_index, total_us=total_us,
            fallback_count=fallback_count, tier_transitions=0,
            missed_tensors=tuple(missed), pool_used_after_token=self.tier_controller.pool_used_bytes,
        ))
        self._token_index += 1
