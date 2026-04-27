"""
substrate.calibration — Offline calibration tooling for Substrate's quality estimator.

This package builds the (layer × op_kind × precision) → expected_loss table that
HybridQualityEstimator consumes. The pipeline:

    1. Load a model via a CalibrationBackend (mlx-lm in production; synthetic in tests).
    2. Capture full-precision per-layer activations on a calibration corpus.
    3. For each (layer_id, op_kind, precision) cell, ablate that op to the lower
       precision while keeping everything else FP, run the corpus, and measure
       output divergence from the FP reference.
    4. Aggregate measurements into a CalibrationOutput and serialize.

The package is importable without MLX. Only the MLXLMBackend requires the
optional [calibration] extra (mlx + mlx-lm).
"""

from substrate.calibration.backend import (
    ActivationCapture,
    CalibrationBackend,
    OpAblation,
)
from substrate.calibration.metrics import (
    DivergenceMetric,
    cosine_distance,
    relative_l2,
)
from substrate.calibration.schema import (
    CalibrationCell,
    CalibrationOutput,
    RunConfig,
    write_calibration_run,
)

__all__ = [
    "ActivationCapture",
    "CalibrationBackend",
    "CalibrationCell",
    "CalibrationOutput",
    "DivergenceMetric",
    "OpAblation",
    "RunConfig",
    "cosine_distance",
    "relative_l2",
    "write_calibration_run",
]
