"""
substrate.calibration.activation_dump — Save paired activations for probe training.

Probe training needs paired (FP_reference, ablated) activations per
(op, precision) cell, with sample-level alignment so row N in `fp16.npy`
corresponds to the SAME corpus sequence as row N in `4bit.npy`.

This module is the writer side. The reader is scripts/train_probe.py.

Layout:

    {run_dir}/
        activations/
            layer_0.attention/
                meta.json
                fp16.npy              # [num_samples, hidden_dim]
                bits_2.npy            # [num_samples, hidden_dim]
                bits_3.npy
                bits_4.npy
                bits_6.npy
            layer_0.mlp_dense/
                meta.json
                fp16.npy
                bits_2.npy
                ...
            ...

meta.json per op:
    {
        "op_id": "layer_0.attention",
        "layer_id": 0,
        "op_kind": "attention",
        "hidden_dim": 1536,
        "num_samples": 32,
        "precisions_present": [2, 3, 4, 6],
        "sample_shape": [32, 1536],
        "dtype": "float32"
    }

Format choice: NumPy .npy, fp32. Smaller than fp16 conversion overhead
in NumPy, and probe training is happiest with fp32 anyway. ~50 MB total
for Qwen2.5-1.5B at 32 sequences × 56 ops × 5 representations × 1536 dims.

Sample alignment:
    The runner processes sequences in order. We accumulate per-(op, precision)
    arrays in memory, one row per sequence, in the order sequences arrive.
    Because the runner's outer loop is sequence-major, we need accumulators
    that grow by one row per sequence per (op, precision).

    Row N in fp16.npy and row N in bits_4.npy are GUARANTEED to be from the
    same sequence index N. This is a load-bearing invariant; probe training
    relies on it for pairing.

Memory model:
    Accumulate as a list-of-1D-arrays during the sweep, stack and write
    at the end. Memory peak ~= total dump size, ~50 MB for Qwen2.5-1.5B.
    For larger models (Qwen3-MoE-30B), this approaches GB; the dumper
    should switch to memmap-backed streaming. Out of scope for v0.1.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# Sentinel name for the FP reference file. We special-case this rather
# than using e.g. "bits_16" because fp16 is conceptually distinct from
# the ablation precisions: it's the reference, not an ablation.
_FP_FILENAME = "fp16.npy"


def _bits_filename(bits: int) -> str:
    return f"bits_{bits}.npy"


# ---------------------------------------------------------------------------
# In-memory accumulator. One per op.
# ---------------------------------------------------------------------------
@dataclass
class _OpAccumulator:
    """
    Accumulates pooled activation rows per (precision) bucket plus the
    fp16 reference. Each row is a 1D vector of length hidden_dim. Rows
    are appended in sequence-arrival order.

    Invariant: at any point, len(fp_rows) >= len(precision_rows[any bits]).
    The runner always captures the FP reference before any ablation
    in a given sequence, so the FP list grows first.
    """
    op_id: str
    layer_id: int
    op_kind: str
    hidden_dim: int | None = None
    fp_rows: list = field(default_factory=list)
    precision_rows: dict[int, list] = field(default_factory=dict)

    def record_fp(self, vector) -> None:
        """Append the FP reference activation for one sequence."""
        self._record(self.fp_rows, vector)

    def record_precision(self, bits: int, vector) -> None:
        """Append the ablated activation for one sequence at a precision."""
        bucket = self.precision_rows.setdefault(bits, [])
        self._record(bucket, vector)

    def _record(self, bucket: list, vector) -> None:
        # Determine hidden_dim from the first row we see.
        # vector can be any iterable of floats (the runner gives us
        # the result of ActivationCapture.flatten()).
        flat = list(vector)
        if self.hidden_dim is None:
            self.hidden_dim = len(flat)
        elif self.hidden_dim != len(flat):
            raise ValueError(
                f"op {self.op_id}: hidden_dim mismatch — saw {self.hidden_dim} "
                f"earlier, got {len(flat)} now."
            )
        bucket.append(flat)


# ---------------------------------------------------------------------------
# Public dumper.
# ---------------------------------------------------------------------------
class ActivationDump:
    """
    Holds in-memory activation accumulators across an entire calibration
    sweep, then writes them to disk in one pass when the sweep completes.

    The dumper is opt-in: the runner only constructs and feeds it when the
    user passed --save-activations. Default calibration runs incur zero
    extra memory or disk cost.
    """

    def __init__(self) -> None:
        self._accumulators: dict[str, _OpAccumulator] = {}

    def register_op(self, op_id: str, layer_id: int, op_kind: str) -> None:
        """
        Pre-create an accumulator. Allows the runner to declare which ops
        will be observed up front; useful for asserting completeness later.
        """
        if op_id not in self._accumulators:
            self._accumulators[op_id] = _OpAccumulator(
                op_id=op_id, layer_id=layer_id, op_kind=op_kind,
            )

    def record_fp(self, op_id: str, layer_id: int, op_kind: str, vector) -> None:
        """Record one FP reference vector for an op (one sequence)."""
        if op_id not in self._accumulators:
            self.register_op(op_id, layer_id, op_kind)
        self._accumulators[op_id].record_fp(vector)

    def record_precision(
        self,
        op_id: str, layer_id: int, op_kind: str,
        bits: int, vector,
    ) -> None:
        """Record one ablated vector for an op at a precision (one sequence)."""
        if op_id not in self._accumulators:
            self.register_op(op_id, layer_id, op_kind)
        self._accumulators[op_id].record_precision(bits, vector)

    def write(self, run_dir: Path | str) -> dict[str, Any]:
        """
        Materialize all accumulators to disk under run_dir/activations/.

        Returns a summary dict for inclusion in metrics_summary.json so
        consumers can quickly see what was saved without crawling the
        directory.
        """
        # Lazy import: numpy isn't a hard dep of substrate.calibration's
        # core modules. activation_dump pulls it in only when actually used.
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "ActivationDump.write requires numpy. "
                "Install with: pip install 'substrate[calibration]' or pip install numpy"
            ) from e

        root = Path(run_dir) / "activations"
        root.mkdir(parents=True, exist_ok=True)

        summary: dict[str, Any] = {
            "format": "npy_v1",
            "ops": [],
            "total_bytes_estimate": 0,
        }

        for op_id, acc in self._accumulators.items():
            op_dir = root / op_id
            op_dir.mkdir(parents=True, exist_ok=True)

            # Verify alignment: every precision bucket should have the
            # same number of rows as fp_rows. If not, something went
            # wrong upstream; warn loudly.
            fp_n = len(acc.fp_rows)
            for bits, rows in acc.precision_rows.items():
                if len(rows) != fp_n:
                    log.warning(
                        "Activation alignment mismatch for op %s @ %d-bit: "
                        "%d ablated rows vs %d FP rows. Probe training will "
                        "use the lesser count.",
                        op_id, bits, len(rows), fp_n,
                    )

            # FP reference.
            if fp_n == 0:
                log.warning("op %s: no FP rows recorded; skipping", op_id)
                continue
            fp_arr = np.array(acc.fp_rows, dtype=np.float32)
            fp_path = op_dir / _FP_FILENAME
            np.save(fp_path, fp_arr)
            op_bytes = fp_arr.nbytes

            # Each precision.
            precisions_present: list[int] = []
            for bits in sorted(acc.precision_rows.keys()):
                rows = acc.precision_rows[bits]
                if not rows:
                    continue
                # Truncate to fp_n if alignment was off — guarantee row N
                # corresponds across files for any reader.
                rows = rows[:fp_n]
                arr = np.array(rows, dtype=np.float32)
                p_path = op_dir / _bits_filename(bits)
                np.save(p_path, arr)
                op_bytes += arr.nbytes
                precisions_present.append(bits)

            # meta.json per op.
            meta = {
                "op_id": acc.op_id,
                "layer_id": acc.layer_id,
                "op_kind": acc.op_kind,
                "hidden_dim": acc.hidden_dim,
                "num_samples": fp_n,
                "precisions_present": precisions_present,
                "sample_shape": [fp_n, acc.hidden_dim],
                "dtype": "float32",
            }
            meta_path = op_dir / "meta.json"
            tmp = meta_path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
            os.replace(tmp, meta_path)

            summary["ops"].append({
                "op_id": acc.op_id,
                "num_samples": fp_n,
                "precisions": precisions_present,
                "bytes": op_bytes,
            })
            summary["total_bytes_estimate"] += op_bytes

        log.info(
            "ActivationDump.write: %d ops -> %s (%.1f MB)",
            len(summary["ops"]), root,
            summary["total_bytes_estimate"] / (1024 * 1024),
        )
        return summary
