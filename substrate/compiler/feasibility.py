from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from substrate.compiler.planner import Budget, ModelProfile, OpProfile

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

def check_feasibility(profile, budget) -> FeasibilityReport:
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

    ceiling_quality = sum(op.best_quality_loss for op in profile.ops)
    if ceiling_quality > budget.quality_loss_cap:
        return FeasibilityReport(
            feasible=False,
            binding_axis=BindingAxis.QUALITY,
            reason=(
                f"Even at maximum precision, quality loss ({ceiling_quality:.4f}) "
                f"exceeds the cap ({budget.quality_loss_cap:.4f})."
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
