"""
substrate.calibration.runner — Orchestrates calibration sweeps.

The runner is backend-agnostic. It:

    1. Asks the backend to discover ops.
    2. Encodes the corpus.
    3. For each sequence:
        a. Captures FP reference activations (one forward pass).
        b. For each (op, precision) cell, captures the ablated activation
           (one forward pass per cell), measures divergence, accumulates.
    4. Aggregates per-cell stats into a CalibrationOutput.

Single-threaded by design. The MLX backend is not thread-safe across
forward passes, and the calibration cost is dominated by GPU time, not
Python orchestration. Adding parallelism is a v0.2 concern.

Cost model: for N sequences, M ops, P precisions, the runner does
N * (1 + M*P) forward passes. For Qwen2.5-1.5B (28 layers, 2 ops/layer = 56)
with P=4 precisions and N=64 sequences, that's 64 * (1 + 224) = 14,400 forward
passes. At ~50ms per pass on M-series silicon, that's about 12 minutes.
Acceptable for a one-shot offline run.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from substrate.calibration.backend import (
    ActivationCapture,
    CalibrationBackend,
    OpAblation,
    OpDescriptor,
)
from substrate.calibration.metrics import (
    DivergenceMetric,
    aggregate_losses,
    get_metric,
)
from substrate.calibration.schema import (
    CalibrationCell,
    CalibrationOutput,
    RunConfig,
    utc_timestamp,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runner config — what to test, with what.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunnerOptions:
    """
    User-facing knobs for a calibration run. Defaults are chosen so a Qwen2.5-1.5B
    sweep finishes in 10-20 minutes on M-series Apple silicon.
    """
    precisions: tuple[int, ...] = (2, 3, 4, 6)
    op_kinds: tuple[str, ...] = ("attention", "mlp_dense", "moe_dispatch")
    metric_name: str = "cosine_distance"
    max_sequences: int = 64
    sequence_length: int = 512
    progress_every: int = 1   # Log every N sequences

    def __post_init__(self) -> None:
        if not self.precisions:
            raise ValueError("precisions must be non-empty")
        if self.max_sequences <= 0:
            raise ValueError("max_sequences must be > 0")
        if self.sequence_length < 8:
            raise ValueError("sequence_length must be >= 8")


# ---------------------------------------------------------------------------
# Per-cell accumulator. Built up across sequences, finalized at the end.
# ---------------------------------------------------------------------------
class _CellAccumulator:
    __slots__ = ("layer_id", "op_kind", "precision_bits", "raw_losses")

    def __init__(self, layer_id: int, op_kind: str, precision_bits: int):
        self.layer_id = layer_id
        self.op_kind = op_kind
        self.precision_bits = precision_bits
        self.raw_losses: list[float] = []

    def add(self, loss: float) -> None:
        self.raw_losses.append(loss)

    def finalize(self) -> CalibrationCell:
        mean, variance, n, mn, mx, median = aggregate_losses(self.raw_losses)
        return CalibrationCell(
            layer_id=self.layer_id,
            op_kind=self.op_kind,
            precision_bits=self.precision_bits,
            expected_loss=mean,
            variance=variance,
            samples=n,
            min_loss=mn,
            max_loss=mx,
            median_loss=median,
        )


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------
class CalibrationRunner:
    """
    Drives a calibration backend through a corpus and accumulates per-cell
    divergence statistics.

    Usage:

        runner = CalibrationRunner(backend, options)
        output = runner.run(corpus_text, corpus_path)
        # output is a CalibrationOutput; pass to write_calibration_run.
    """

    def __init__(
        self,
        backend: CalibrationBackend,
        options: RunnerOptions | None = None,
    ) -> None:
        self.backend = backend
        self.options = options or RunnerOptions()
        self._metric: DivergenceMetric = get_metric(self.options.metric_name)

    def run(self, corpus_text: str, corpus_path: str | Path) -> CalibrationOutput:
        started_at = time.time()
        started_iso = utc_timestamp()

        # Filter ops to the user-requested kinds.
        all_ops = self.backend.discover_ops()
        ops = tuple(op for op in all_ops if op.op_kind in self.options.op_kinds)
        if not ops:
            raise RuntimeError(
                f"No ops match requested kinds {self.options.op_kinds}. "
                f"Backend discovered: {sorted({op.op_kind for op in all_ops})}"
            )
        log.info(
            "Calibrating %d ops across %d precisions (%d total cells)",
            len(ops), len(self.options.precisions),
            len(ops) * len(self.options.precisions),
        )

        # Build accumulators: one per (layer, op_kind, precision) cell.
        accumulators: dict[tuple[int, str, int], _CellAccumulator] = {}
        for op in ops:
            for bits in self.options.precisions:
                key = (op.layer_id, op.op_kind, bits)
                accumulators[key] = _CellAccumulator(op.layer_id, op.op_kind, bits)

        # Encode corpus.
        sequences = self.backend.encode_corpus(
            corpus_text,
            max_sequences=self.options.max_sequences,
            sequence_length=self.options.sequence_length,
        )
        if not sequences:
            raise RuntimeError(
                "Corpus produced zero usable sequences. Check the file content "
                "and sequence_length."
            )
        log.info(
            "Encoded corpus into %d sequences of up to %d tokens",
            len(sequences), self.options.sequence_length,
        )

        # The main sweep.
        for seq_idx, sequence in enumerate(sequences):
            self._run_one_sequence(seq_idx, sequence, ops, accumulators)
            if (seq_idx + 1) % self.options.progress_every == 0:
                log.info(
                    "  sequence %d/%d done", seq_idx + 1, len(sequences),
                )

        # Finalize cells.
        cells = tuple(acc.finalize() for acc in accumulators.values())
        elapsed = time.time() - started_at
        log.info(
            "Calibration complete: %d cells, %d total samples, %.1fs",
            len(cells), sum(c.samples for c in cells), elapsed,
        )

        config = RunConfig(
            model_id=self.backend.model_id,
            backend=self.backend.name,
            backend_version=self.backend.version,
            corpus_path=str(corpus_path),
            corpus_sha256=hashlib.sha256(corpus_text.encode("utf-8")).hexdigest(),
            num_sequences=len(sequences),
            sequence_length=self.options.sequence_length,
            precisions_tested=tuple(self.options.precisions),
            op_kinds_tested=tuple(self.options.op_kinds),
            divergence_metric=self.options.metric_name,
            started_at_utc=started_iso,
            duration_seconds=elapsed,
        )
        return CalibrationOutput(config=config, cells=cells)

    # ------------------------------------------------------------------
    # Per-sequence inner loop.
    # ------------------------------------------------------------------
    def _run_one_sequence(
        self,
        seq_idx: int,
        sequence: Sequence[int],
        ops: tuple[OpDescriptor, ...],
        accumulators: dict[tuple[int, str, int], _CellAccumulator],
    ) -> None:
        # FP reference: one forward pass.
        try:
            ref_captures = self.backend.capture_reference(sequence)
        except Exception:
            log.exception(
                "Reference capture failed on sequence %d; skipping", seq_idx,
            )
            return

        # Per (op, precision) ablation: one forward pass each.
        for op in ops:
            ref = ref_captures.get(op.op_id)
            if ref is None:
                # Backend didn't produce a reference for this op. Skip
                # but log loudly — this is usually a backend bug.
                log.warning(
                    "Backend did not capture reference for op %s; skipping",
                    op.op_id,
                )
                continue
            ref_flat = ref.flatten()

            for bits in self.options.precisions:
                ablation = OpAblation(
                    op_id=op.op_id,
                    layer_id=op.layer_id,
                    op_kind=op.op_kind,
                    precision_bits=bits,
                )
                try:
                    cand = self.backend.ablate_op(sequence, ablation)
                except Exception:
                    log.exception(
                        "Ablation failed: op=%s bits=%d seq=%d; skipping cell",
                        op.op_id, bits, seq_idx,
                    )
                    continue
                cand_flat = cand.flatten()
                try:
                    loss = self._metric(ref_flat, cand_flat)
                except Exception:
                    log.exception(
                        "Metric failed: op=%s bits=%d seq=%d; skipping",
                        op.op_id, bits, seq_idx,
                    )
                    continue
                accumulators[(op.layer_id, op.op_kind, bits)].add(loss)
