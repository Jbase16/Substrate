from __future__ import annotations
import logging, threading, time
from collections import deque
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue
from typing import Callable, Mapping
from substrate.compiler.ir import PrefetchRequest, ScheduledOp, TensorMetadata
from substrate.runtime.residency import ResidencyManager

log = logging.getLogger(__name__)

@dataclass(order=True)
class _PrefetchJob:
    sort_key: tuple
    request: PrefetchRequest = field(compare=False)
    deadline_op_index: int = field(compare=False)
    enqueued_at_ns: int = field(compare=False)

@dataclass
class PrefetchOutcome:
    tensor_id: str
    success: bool
    detail: str
    elapsed_us: int

class PrefetchScheduler:
    def __init__(self, ops, catalog, residency, sustained_bandwidth_bps, on_outcome=None):
        self._ops = ops
        self._op_index = {op.op_id: i for i, op in enumerate(ops)}
        self._catalog = catalog
        self._residency = residency
        self._bandwidth = sustained_bandwidth_bps
        self._on_outcome = on_outcome
        self._queue: PriorityQueue = PriorityQueue()
        self._sequence = 0
        self._enqueue_lock = threading.Lock()
        self._bw_window_ns = 1_000_000_000
        self._bw_history: deque = deque()
        self._bw_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker = None

    def start(self):
        if self._worker is not None:
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run, name="substrate-prefetch", daemon=True)
        self._worker.start()

    def stop(self, timeout=1.0):
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout)
            self._worker = None

    def on_op_start(self, op: ScheduledOp):
        for pf in op.prefetch:
            if not self._residency.is_resident(pf.tensor.tensor_id):
                self._enqueue(pf)

    def on_op_end(self, op: ScheduledOp):
        pass

    def wait_until_resident(self, tensor_ids, timeout_us):
        deadline_ns = time.perf_counter_ns() + timeout_us * 1000
        missed = []
        for tid in tensor_ids:
            while not self._residency.is_resident(tid):
                remaining_ns = deadline_ns - time.perf_counter_ns()
                if remaining_ns <= 0:
                    missed.append(tid)
                    break
                time.sleep(min(remaining_ns / 1e9, 0.001))
        return tuple(missed)

    def _enqueue(self, pf):
        with self._enqueue_lock:
            self._sequence += 1
            seq = self._sequence
        job = _PrefetchJob(
            sort_key=(self._op_index[pf.deadline_before], -pf.priority, seq),
            request=pf, deadline_op_index=self._op_index[pf.deadline_before],
            enqueued_at_ns=time.perf_counter_ns(),
        )
        self._queue.put(job)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.05)
            except Empty:
                continue
            self._service_job(job)

    def _service_job(self, job):
        tid = job.request.tensor.tensor_id
        if self._residency.is_resident(tid):
            self._emit_outcome(tid, True, "already_resident", 0)
            return
        meta = self._catalog[tid]
        now_ns = time.perf_counter_ns()
        if not self._reserve_bandwidth(meta.bytes_on_ssd, now_ns):
            self._enqueue(job.request)
            return
        read_secs = meta.bytes_on_ssd / max(1, self._bandwidth)
        time.sleep(read_secs)
        try:
            self._residency.load(tid)
        except RuntimeError as e:
            self._emit_outcome(tid, False, f"ram_exceeded:{e}", int(read_secs * 1e6))
            return
        elapsed_us = (time.perf_counter_ns() - job.enqueued_at_ns) // 1000
        self._emit_outcome(tid, True, "loaded", elapsed_us)

    def _reserve_bandwidth(self, bytes_count, now_ns):
        with self._bw_lock:
            cutoff = now_ns - self._bw_window_ns
            while self._bw_history and self._bw_history[0][0] < cutoff:
                self._bw_history.popleft()
            current_total = sum(b for _, b in self._bw_history)
            cap = self._bandwidth * (self._bw_window_ns / 1e9)
            if current_total + bytes_count > cap:
                return False
            self._bw_history.append((now_ns, bytes_count))
            return True

    def _emit_outcome(self, tensor_id, success, detail, elapsed_us):
        if self._on_outcome is None:
            return
        try:
            self._on_outcome(PrefetchOutcome(tensor_id, success, detail, elapsed_us))
        except Exception:
            log.exception("prefetch outcome callback raised")
