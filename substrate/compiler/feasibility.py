"""
substrate.compiler.feasibility — Pre-flight feasibility checks.

Four binding axes, evaluated in this order:

    MEMORY    — Even the skeleton-only configuration exceeds the RAM budget.
    BANDWIDTH — SSD throughput cannot serve the residual stream in real time.
    LATENCY   — Best-case op latency sums above the target tokens-per-second.
    QUALITY   — Even at maximum precision, predicted loss exceeds the cap.

The check is grounded in the profile metadata and (for the QUALITY axis)
optionally in a real QualityEstimator. Without an estimator, the QUALITY
check uses the profile's per-op `best_quality_loss` surrogate as the
ceiling, which is conservative but unrealistic. With an estimator, the
ceiling is computed by querying the estimator at all-NEAR_FP16 — the same
ceiling the planner's post-fill check would discover.

Calling check_feasibility from the planner ensures both pre-flight and
post-fill quality checks agree (they query the same estimator). Calling
it from the CLI without an estimator (`feasibility` subcommand) gives a
quick conservative answer without requiring a calibration file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from substrate.compiler.planner import Budget, ModelProfile
    from substrate.compiler.quality import QualityEstimator


class BindingAxis(str, Enum):
    NONE = "none"
    MEMORY = "memory"
    BANDWIDTH = "bandwidth"
    LATENCY = "latency"
    QUALITY = "quality"


@dataclass(frozen=True)
class FeasibilityReport:
    feasible: bool
    binding_axis: BindingAxis
    reason: str
    relax_options: dict = field(default_factory=dict)
    floor_resident_bytes: int = 0
    floor_ssd_bandwidth_bps: int = 0
    floor_compute_us_per_token: int = 0
    ceiling_quality_loss: float = 0.0

    def to_dict(self) -> dict:
        return {
            "status": "feasible" if self.feasible else "infeasible",
            "binding_axis": self.binding_axis.value,
            "reason": self.reason,
            "relax_options": dict(self.relax_options),
            "diagnostics": {
                "floor_resident_bytes": self.floor_resident_bytes,
                "floor_ssd_bandwidth_bps": self.floor_ssd_bandwidth_bps,
                "floor_compute_us_per_token": self.floor_compute_us_per_token,
                "ceiling_quality_loss": self.ceiling_quality_loss,
            },
        }


class InfeasibleBudgetError(Exception):
    def __init__(self, report: FeasibilityReport):
        self.report = report
        super().__init__(report.reason)


def check_feasibility(
    profile: "ModelProfile",
    budget: "Budget",
    quality_estimator: "QualityEstimator | None" = None,
) -> FeasibilityReport:
    """
    Run all four feasibility axes. Returns a FeasibilityReport.

    The optional quality_estimator is queried at all-NEAR_FP16 to compute
    the realistic quality ceiling. Without it, the check falls back to
    `sum(op.best_quality_loss for op in profile.ops)` — a surrogate that
    is conservative but disconnected from real measurement. Always pass
    the estimator when one is available; the surrogate is for tests and
    quick CLI feasibility checks against profiles alone.
    """
    # MEMORY axis.
    floor_ram = (
        profile.embedding_bytes
        + profile.lm_head_bytes
        + profile.runtime_overhead_bytes
        + sum(op.skeleton_bytes for op in profile.ops)
    )
    if floor_ram > budget.max_ram_bytes:
        needed_gb = floor_ram / 1e9
        have_gb = budget.max_ram_bytes / 1e9
        return FeasibilityReport(
            feasible=False,
            binding_axis=BindingAxis.MEMORY,
            reason=(
                f"Skeleton-only configuration requires {needed_gb:.2f} GB resident, "
                f"budget allows {have_gb:.2f} GB. The model's compressed skeleton "
                f"alone does not fit."
            ),
            relax_options={
                "max_ram_bytes": f">= {floor_ram} ({needed_gb:.2f} GB)",
                "model": "use a smaller or more aggressively quantized model",
            },
            floor_resident_bytes=floor_ram,
        )

    # BANDWIDTH axis.
    residual_bytes_per_token = sum(
        op.minimum_residual_bytes_per_token for op in profile.ops
    )
    target_tps = budget.target_tokens_per_second or 1.0
    required_bandwidth_bps = int(residual_bytes_per_token * target_tps)

    if required_bandwidth_bps > budget.sustained_ssd_bw_bytes_per_sec:
        needed_gbps = required_bandwidth_bps / 1e9
        have_gbps = budget.sustained_ssd_bw_bytes_per_sec / 1e9
        achievable_tps = (
            budget.sustained_ssd_bw_bytes_per_sec / max(1, residual_bytes_per_token)
        )
        return FeasibilityReport(
            feasible=False,
            binding_axis=BindingAxis.BANDWIDTH,
            reason=(
                f"Residual stream requires {needed_gbps:.2f} GB/s sustained read "
                f"to meet {target_tps:.1f} tok/s. SSD budget is {have_gbps:.2f} GB/s."
            ),
            relax_options={
                "target_tokens_per_second": f"<= {achievable_tps:.2f}",
                "sustained_ssd_bw_bytes_per_sec": f">= {required_bandwidth_bps}",
                "model": "use a model with smaller residual planes",
            },
            floor_resident_bytes=floor_ram,
            floor_ssd_bandwidth_bps=required_bandwidth_bps,
        )

    # LATENCY axis.
    floor_compute_us = sum(op.skeleton_compute_us for op in profile.ops)
    if budget.target_tokens_per_second is not None:
        target_us = int(1e6 / budget.target_tokens_per_second)
        if floor_compute_us > target_us:
            achievable_tps = 1e6 / floor_compute_us
            return FeasibilityReport(
                feasible=False,
                binding_axis=BindingAxis.LATENCY,
                reason=(
                    f"Skeleton-only forward pass takes {floor_compute_us / 1000:.1f} ms "
                    f"per token; target {target_us / 1000:.1f} ms is below the floor."
                ),
                relax_options={
                    "target_tokens_per_second": f"<= {achievable_tps:.2f}",
                    "model": "use a smaller model or different hardware",
                },
                floor_resident_bytes=floor_ram,
                floor_compute_us_per_token=floor_compute_us,
            )

    # QUALITY axis. Use the estimator if provided; otherwise the surrogate.
    ceiling_quality = _compute_quality_ceiling(profile, quality_estimator)
    if ceiling_quality > budget.quality_loss_cap:
        ceiling_source = "calibrated" if quality_estimator is not None else "surrogate"
        return FeasibilityReport(
            feasible=False,
            binding_axis=BindingAxis.QUALITY,
            reason=(
                f"Even at maximum precision, predicted quality loss "
                f"({ceiling_quality:.4f}, {ceiling_source}) exceeds the cap "
                f"({budget.quality_loss_cap:.4f})."
            ),
            relax_options={
                "quality_loss_cap": f">= {ceiling_quality:.4f}",
                "model": "use a model with better quantization characteristics",
            },
            floor_resident_bytes=floor_ram,
            ceiling_quality_loss=ceiling_quality,
        )

    return FeasibilityReport(
        feasible=True,
        binding_axis=BindingAxis.NONE,
        reason="All axes within budget.",
        floor_resident_bytes=floor_ram,
        floor_ssd_bandwidth_bps=required_bandwidth_bps,
        floor_compute_us_per_token=floor_compute_us,
        ceiling_quality_loss=ceiling_quality,
    )


def _compute_quality_ceiling(
    profile: "ModelProfile",
    estimator: "QualityEstimator | None",
) -> float:
    """
    Compute the realistic best-case quality loss.

    With an estimator: query at all-NEAR_FP16. This is what the planner's
    post-fill check uses, so feasibility and the planner agree on the
    ceiling. The estimator's clamped expected_loss is the answer.

    Without an estimator: fall back to the profile's surrogate
    sum(op.best_quality_loss). Conservative but disconnected from real
    measurement.
    """
    if estimator is None:
        return sum(op.best_quality_loss for op in profile.ops)

    # Build all-NEAR_FP16 assignment. Import locally to avoid a top-level
    # circular import (planner imports feasibility).
    from substrate.compiler.planner import Precision

    assignment = {
        (op.layer_id, op.op_kind): Precision.NEAR_FP16 for op in profile.ops
    }
    return estimator.estimate(assignment).expected_loss
