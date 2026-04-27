"""
substrate.calibration.schema — Output format for calibration runs.

A calibration run produces a directory:

    calibration/
      {model_id}/
        {timestamp}/
          calibration.json     # The (layer, op_kind, precision) -> stats table
          run_config.json      # What was run, with what settings
          metrics_summary.json # Aggregate stats across the run

The calibration.json format matches what HybridQualityEstimator expects via
substrate.compiler.quality.CalibrationEntry, plus richer metadata.

This module is the canonical schema definition. Any calibration backend (mlx-lm,
synthetic, future PyTorch backend) must produce data conforming to this schema.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# Output cell. One row in the final table.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CalibrationCell:
    """
    Stats for a single (layer_id, op_kind, precision) cell.

    op_kind values must match substrate.compiler.ir.OpKind values:
        attention | mlp_dense | moe_router | moe_dispatch | embedding | norm | lm_head

    precision_bits is the effective bit-width used for ablation. Substrate's
    Precision enum maps to: 2, 3, 4, 6, 16. Future precisions (5, 8) can be
    added without breaking the schema.

    expected_loss is the mean divergence between the FP reference output and
    the ablated output, computed via the chosen DivergenceMetric. Units depend
    on the metric — for cosine_distance, it's [0, 2]; for relative_l2, it's
    a unitless ratio. The metric used is recorded in RunConfig.

    variance is the sample variance over the calibration corpus. Used by
    HybridQualityEstimator for confidence intervals.

    samples is the number of activation pairs measured for this cell. Higher
    samples = tighter variance estimate.
    """
    layer_id: int
    op_kind: str
    precision_bits: int
    expected_loss: float
    variance: float
    samples: int

    # Optional diagnostics. Useful for debugging calibration quality but not
    # consumed by the estimator.
    min_loss: float = 0.0
    max_loss: float = 0.0
    median_loss: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Run configuration. Records what was run so results are reproducible.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    model_id: str
    backend: str                          # E.g. "mlx-lm", "synthetic"
    backend_version: str                  # Library version of the backend
    corpus_path: str                      # Absolute path to the corpus file
    corpus_sha256: str                    # Hash of the corpus contents
    num_sequences: int
    sequence_length: int
    precisions_tested: tuple[int, ...]    # E.g. (2, 3, 4, 6)
    op_kinds_tested: tuple[str, ...]      # E.g. ("attention", "mlp_dense")
    divergence_metric: str                # "cosine_distance" | "relative_l2"
    started_at_utc: str                   # ISO 8601
    duration_seconds: float
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["precisions_tested"] = list(d["precisions_tested"])
        d["op_kinds_tested"] = list(d["op_kinds_tested"])
        return d


# ---------------------------------------------------------------------------
# The full output. Assembled by the runner, written by write_calibration_run.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CalibrationOutput:
    config: RunConfig
    cells: tuple[CalibrationCell, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "config": self.config.to_dict(),
            "cells": [c.to_dict() for c in self.cells],
        }

    def summary(self) -> dict[str, Any]:
        """Aggregate stats across all cells. Used for metrics_summary.json."""
        if not self.cells:
            return {"num_cells": 0}
        losses = [c.expected_loss for c in self.cells]
        per_kind: dict[str, list[float]] = {}
        per_precision: dict[int, list[float]] = {}
        for c in self.cells:
            per_kind.setdefault(c.op_kind, []).append(c.expected_loss)
            per_precision.setdefault(c.precision_bits, []).append(c.expected_loss)
        return {
            "num_cells": len(self.cells),
            "total_samples": sum(c.samples for c in self.cells),
            "global_mean_loss": sum(losses) / len(losses),
            "global_max_loss": max(losses),
            "global_min_loss": min(losses),
            "mean_loss_by_op_kind": {
                k: sum(v) / len(v) for k, v in per_kind.items()
            },
            "mean_loss_by_precision": {
                str(k): sum(v) / len(v) for k, v in per_precision.items()
            },
        }


# ---------------------------------------------------------------------------
# Output location helpers.
# ---------------------------------------------------------------------------
def utc_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp like '2026-04-27T214502Z'."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H%M%SZ")


def make_run_dir(output_root: Path | str, model_id: str) -> Path:
    """
    Create calibration/{model_id}/{timestamp}/ and return its path.

    model_id is sanitized: '/' becomes '__' so HuggingFace paths like
    'mlx-community/Qwen2.5-1.5B-Instruct-4bit' produce a clean directory name.
    """
    safe_id = model_id.replace("/", "__").replace(" ", "_")
    run_dir = Path(output_root) / safe_id / utc_timestamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_calibration_run(output: CalibrationOutput, run_dir: Path) -> dict[str, Path]:
    """
    Write calibration.json + run_config.json + metrics_summary.json to run_dir.

    Returns a dict of {filename: written_path} so callers can log/return paths.

    Atomic writes via temp files: we write to .tmp and rename. If anything
    fails midway, the run directory contains no half-written files.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "calibration.json": run_dir / "calibration.json",
        "run_config.json": run_dir / "run_config.json",
        "metrics_summary.json": run_dir / "metrics_summary.json",
    }

    payloads = {
        "calibration.json": output.to_dict(),
        "run_config.json": output.config.to_dict(),
        "metrics_summary.json": output.summary(),
    }

    for name, target in paths.items():
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(payloads[name], f, indent=2, sort_keys=True)
        os.replace(tmp, target)

    return paths


def load_calibration(calibration_json_path: Path | str) -> CalibrationOutput:
    """
    Load a previously-written calibration.json back into a CalibrationOutput.

    Used by the planner-side adapter (substrate.calibration.adapter) to feed
    real calibration data into HybridQualityEstimator.
    """
    path = Path(calibration_json_path)
    with open(path) as f:
        data = json.load(f)
    if data.get("format_version") != 1:
        raise ValueError(
            f"Calibration file {path}: unknown format_version "
            f"{data.get('format_version')}, expected 1"
        )
    config_data = data["config"]
    config = RunConfig(
        model_id=config_data["model_id"],
        backend=config_data["backend"],
        backend_version=config_data["backend_version"],
        corpus_path=config_data["corpus_path"],
        corpus_sha256=config_data["corpus_sha256"],
        num_sequences=config_data["num_sequences"],
        sequence_length=config_data["sequence_length"],
        precisions_tested=tuple(config_data["precisions_tested"]),
        op_kinds_tested=tuple(config_data["op_kinds_tested"]),
        divergence_metric=config_data["divergence_metric"],
        started_at_utc=config_data["started_at_utc"],
        duration_seconds=config_data["duration_seconds"],
        extra=config_data.get("extra", {}),
    )
    cells = tuple(
        CalibrationCell(
            layer_id=c["layer_id"],
            op_kind=c["op_kind"],
            precision_bits=c["precision_bits"],
            expected_loss=c["expected_loss"],
            variance=c["variance"],
            samples=c["samples"],
            min_loss=c.get("min_loss", 0.0),
            max_loss=c.get("max_loss", 0.0),
            median_loss=c.get("median_loss", 0.0),
        )
        for c in data["cells"]
    )
    return CalibrationOutput(config=config, cells=cells)
