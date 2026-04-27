"""
substrate.calibration.adapter — Bridge from CalibrationOutput to HybridQualityEstimator.

The estimator wants a Mapping[(layer_id, OpKind, Precision), CalibrationEntry].
The calibration runner produces CalibrationOutput with op_kind as a string and
precision as int bits. This module does the type conversion:

    str op_kind   -> substrate.compiler.ir.OpKind
    int bits      -> substrate.compiler.planner.Precision
    CalibrationCell -> substrate.compiler.quality.CalibrationEntry

Importing this module pulls in the planner. That's fine — anyone using the
adapter is by definition planning with the calibration data.

Cells whose op_kind / precision_bits don't map to known enum values are
skipped with a warning, not raised. Real calibration runs may include op
kinds we haven't taught the planner about yet (e.g. MoE_ROUTER ablation
on a dense model).
"""

from __future__ import annotations

import logging
from typing import Mapping

from substrate.calibration.schema import CalibrationCell, CalibrationOutput
from substrate.compiler.ir import OpKind
from substrate.compiler.planner import Precision
from substrate.compiler.quality import CalibrationEntry, HybridQualityEstimator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mappings between calibration string types and Substrate enums.
# ---------------------------------------------------------------------------
_BITS_TO_PRECISION: dict[int, Precision] = {
    2: Precision.SKELETON_2BIT,
    3: Precision.SKELETON_PLUS_R1,
    4: Precision.SKELETON_PLUS_R2,
    6: Precision.REFINED_6BIT,
    16: Precision.NEAR_FP16,
}


def calibration_to_table(
    output: CalibrationOutput,
) -> dict[tuple[int, OpKind, Precision], CalibrationEntry]:
    """
    Convert a CalibrationOutput to the lookup table HybridQualityEstimator
    expects.

    Cells with unknown op_kinds or unsupported precision_bits are skipped
    with a warning. The returned table is suitable for direct construction
    of HybridQualityEstimator.
    """
    table: dict[tuple[int, OpKind, Precision], CalibrationEntry] = {}
    op_kind_map = {k.value: k for k in OpKind}

    skipped_unknown_kind = 0
    skipped_unknown_bits = 0

    for cell in output.cells:
        kind = op_kind_map.get(cell.op_kind)
        if kind is None:
            skipped_unknown_kind += 1
            continue
        precision = _BITS_TO_PRECISION.get(cell.precision_bits)
        if precision is None:
            skipped_unknown_bits += 1
            continue

        entry = CalibrationEntry(
            layer_id=cell.layer_id,
            op_kind=kind,
            precision=precision,
            expected_loss=cell.expected_loss,
            variance=cell.variance,
            samples=cell.samples,
        )
        table[(cell.layer_id, kind, precision)] = entry

    if skipped_unknown_kind:
        log.warning(
            "Skipped %d calibration cells with unknown op_kind",
            skipped_unknown_kind,
        )
    if skipped_unknown_bits:
        log.warning(
            "Skipped %d calibration cells with unsupported precision_bits",
            skipped_unknown_bits,
        )
    return table


def estimator_from_calibration(
    output: CalibrationOutput,
    confidence_z: float = 1.96,
) -> HybridQualityEstimator:
    """One-liner: turn a CalibrationOutput into a usable estimator."""
    table = calibration_to_table(output)
    return HybridQualityEstimator(table, confidence_z=confidence_z)
