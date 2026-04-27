"""
Substrate — MLX-native execution planning for oversized local models under
hard memory constraints.
"""

from substrate.compiler.feasibility import (
    BindingAxis,
    FeasibilityReport,
    InfeasibleBudgetError,
    check_feasibility,
)
from substrate.compiler.ir import (
    Budget,
    EscalationPolicy,
    EvictRule,
    FallbackPolicy,
    FallbackStrategy,
    OpBundle,
    OpKind,
    PlanBundle,
    PrefetchRequest,
    ScheduledOp,
    TensorMetadata,
    TensorRef,
)
from substrate.compiler.planner import (
    ModelProfile,
    OpProfile,
    Planner,
    PlannerOptions,
    Precision,
)
from substrate.compiler.quality import (
    HybridQualityEstimator,
    QualityEstimate,
    QualityEstimator,
)

__version__ = "0.1.0"

__all__ = [
    "BindingAxis", "Budget", "EscalationPolicy", "EvictRule", "FallbackPolicy",
    "FallbackStrategy", "FeasibilityReport", "HybridQualityEstimator",
    "InfeasibleBudgetError", "ModelProfile", "OpBundle", "OpKind", "OpProfile",
    "PlanBundle", "Planner", "PlannerOptions", "Precision", "PrefetchRequest",
    "QualityEstimate", "QualityEstimator", "ScheduledOp", "TensorMetadata",
    "TensorRef", "check_feasibility",
]
