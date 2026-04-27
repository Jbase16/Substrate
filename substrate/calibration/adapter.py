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

NEAR_FP16 reference rows:
    Calibration runs ablate at lower-than-FP precisions (typically 2/3/4/6
    bits) because FP16 ablation is a no-op — by definition, the FP16 weights
    diverge from themselves by zero. But the estimator needs a row for
    NEAR_FP16 to consider it as an upgrade target. We synthesize one here
    by injecting near-zero entries for every (layer, op_kind) pair seen in
    the calibration. Without this, NEAR_FP16 lookups fall back to the
    estimator's no-data constant (0.05/layer), which makes FP16 look like
    a worse choice than 6-bit and breaks rate-distortion ordering.
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

# Synthetic loss for NEAR_FP16 rows. Not exactly zero because:
#   1. The estimator's QualityEstimate validator requires confidence bounds
#      to be a proper interval, and zero-loss with zero-variance creates
#      degenerate intervals.
#   2. Real fp16 still has rounding error vs fp32 reference. A tiny but
#      non-zero loss reflects this physical truth without distorting
#      ordering.
# 1e-6 is well below any measured 6-bit loss (which is ~3e-5 in practice),
# preserving the precision ordering: fp16 < 6-bit < 4-bit < 3-bit < 2-bit.
_FP16_SYNTHETIC_LOSS = 1e-6
_FP16_SYNTHETIC_VARIANCE = (1e-6) ** 2


def calibration_to_table(
    output: CalibrationOutput,
    inject_fp16_reference: bool = True,
) -> dict[tuple[int, OpKind, Precision], CalibrationEntry]:
    """
    Convert a CalibrationOutput to the lookup table HybridQualityEstimator
    expects.

    Cells with unknown op_kinds or unsupported precision_bits are skipped
    with a warning. If inject_fp16_reference is True (default), synthesize
    NEAR_FP16 rows for every (layer, op_kind) pair observed in the
    calibration. This makes FP16 a usable upgrade target instead of falling
    back to the estimator's no-data constant.
    """
    table: dict[tuple[int, OpKind, Precision], CalibrationEntry] = {}
    op_kind_map = {k.value: k for k in OpKind}

    skipped_unknown_kind = 0
    skipped_unknown_bits = 0
    seen_layer_kinds: set[tuple[int, OpKind]] = set()

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
        seen_layer_kinds.add((cell.layer_id, kind))

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

    # Inject NEAR_FP16 reference rows for every observed (layer, op_kind).
    # Skip if calibration already covered fp16 (e.g. from a future runner
    # version that does measure it explicitly).
    if inject_fp16_reference:
        injected = 0
        for layer_id, kind in seen_layer_kinds:
            key = (layer_id, kind, Precision.NEAR_FP16)
            if key in table:
                continue
            table[key] = CalibrationEntry(
                layer_id=layer_id,
                op_kind=kind,
                precision=Precision.NEAR_FP16,
                expected_loss=_FP16_SYNTHETIC_LOSS,
                variance=_FP16_SYNTHETIC_VARIANCE,
                # samples=1 is the minimum the validator allows. We use it
                # to flag these rows as "synthetic, not measured" if anyone
                # inspects the table later.
                samples=1,
            )
            injected += 1
        if injected > 0:
            log.info(
                "Injected %d synthetic NEAR_FP16 reference rows "
                "(loss=%.0e per cell). FP16 is a no-op vs the FP reference; "
                "real calibration runs measure 2/3/4/6-bit divergence only.",
                injected, _FP16_SYNTHETIC_LOSS,
            )

    return table


def estimator_from_calibration(
    output: CalibrationOutput,
    confidence_z: float = 1.96,
    inject_fp16_reference: bool = True,
) -> HybridQualityEstimator:
    """One-liner: turn a CalibrationOutput into a usable estimator."""
    table = calibration_to_table(output, inject_fp16_reference=inject_fp16_reference)
    return HybridQualityEstimator(table, confidence_z=confidence_z)
