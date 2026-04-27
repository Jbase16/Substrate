from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Mapping, Protocol, TYPE_CHECKING
if TYPE_CHECKING:
    from substrate.compiler.ir import OpKind
    from substrate.compiler.planner import Precision

@dataclass(frozen=True)
class QualityEstimate:
    expected_loss: float
    confidence: tuple[float, float]
    binding_layers: tuple[int, ...]
    risk_reason: str
    per_layer_loss: dict = field(default_factory=dict)

    def __post_init__(self):
        lo, hi = self.confidence
        if not (0.0 <= lo <= self.expected_loss <= hi <= 1.0):
            raise ValueError(f"Invalid QualityEstimate bounds: {lo} <= {self.expected_loss} <= {hi}")

@dataclass(frozen=True)
class CalibrationEntry:
    layer_id: int
    op_kind: object
    precision: object
    expected_loss: float
    variance: float
    samples: int

class QualityEstimator(Protocol):
    def estimate(self, plan_assignments, context_features=None) -> QualityEstimate: ...

class HybridQualityEstimator:
    _MAX_DISAGREEMENT_MULTIPLIER = 4.0

    def __init__(self, calibration, confidence_z=1.96):
        self._calibration = dict(calibration)
        self._confidence_z = confidence_z
        kind_prec_buckets = {}
        for (_layer, kind, prec), entry in self._calibration.items():
            kind_prec_buckets.setdefault((kind, prec), []).append(entry)
        self._kind_prec_avg = {}
        for (kind, prec), entries in kind_prec_buckets.items():
            n = len(entries)
            self._kind_prec_avg[(kind, prec)] = CalibrationEntry(
                layer_id=-1, op_kind=kind, precision=prec,
                expected_loss=sum(e.expected_loss for e in entries) / n,
                variance=max(e.variance for e in entries) * 2.0,
                samples=sum(e.samples for e in entries) // n,
            )

    def estimate(self, plan_assignments, context_features=None):
        ctx = context_features or {}
        per_layer, per_layer_var = {}, {}
        miss_count = 0
        for (layer_id, kind), precision in plan_assignments.items():
            entry = self._calibration.get((layer_id, kind, precision)) or self._kind_prec_avg.get((kind, precision))
            if entry is None:
                miss_count += 1
                base_loss, base_var = 0.05, 0.001
            else:
                base_loss, base_var = entry.expected_loss, entry.variance
            disagreement = ctx.get(f"layer_{layer_id}.disagreement", 0.0)
            multiplier = min(self._MAX_DISAGREEMENT_MULTIPLIER, 1.0 + max(0.0, disagreement))
            per_layer[layer_id] = per_layer.get(layer_id, 0.0) + base_loss * multiplier
            per_layer_var[layer_id] = per_layer_var.get(layer_id, 0.0) + base_var * (multiplier ** 2)
        expected = min(1.0, sum(per_layer.values()))
        std = math.sqrt(sum(per_layer_var.values()))
        lo = max(0.0, expected - self._confidence_z * std)
        hi = min(1.0, expected + self._confidence_z * std)
        ranked = sorted(per_layer.items(), key=lambda kv: kv[1], reverse=True)
        binding = tuple(l for l, v in ranked[:8] if v >= expected * 0.05)
        return QualityEstimate(
            expected_loss=expected, confidence=(lo, hi),
            binding_layers=binding, risk_reason="calibration_baseline",
            per_layer_loss=per_layer,
        )

def stub_calibration_table(num_layers, op_kinds, precisions, sensitive_layers=(), seed=0):
    import random
    rng = random.Random(seed)
    bit_to_base = {2.0: 0.0050, 3.0: 0.0025, 4.0: 0.0012, 6.0: 0.0004, 16.0: 0.0001}
    table = {}
    for layer in range(num_layers):
        for kind in op_kinds:
            for prec in precisions:
                base = bit_to_base.get(prec.effective_bits, 0.005)
                if layer in sensitive_layers and prec.effective_bits <= 3.0:
                    base *= 2.0 + rng.random()
                expected = max(0.0, min(1.0, base * (1.0 + (rng.random() - 0.5) * 0.2)))
                table[(layer, kind, prec)] = CalibrationEntry(
                    layer_id=layer, op_kind=kind, precision=prec,
                    expected_loss=expected, variance=(expected * 0.15) ** 2, samples=512,
                )
    return table
