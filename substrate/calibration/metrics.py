"""
substrate.calibration.metrics — Divergence metrics for calibration.

For v0.1 we use cheap activation-level metrics: cosine distance and relative L2.
These correlate with KL divergence on downstream output distributions but are
~10-100x faster because they don't require replaying the rest of the layer
stack from the ablation point.

Both metrics operate on flat numeric vectors. The MLX backend converts MLX
arrays to Python floats / numpy arrays before calling these. The synthetic
backend uses them directly.

KL divergence is intentionally not implemented here. If we want it later, we
add it as a third metric option without changing the schema — the
DivergenceMetric protocol is the seam.
"""

from __future__ import annotations

import math
from typing import Iterable, Protocol, Sequence


# ---------------------------------------------------------------------------
# Pluggable metric protocol.
# ---------------------------------------------------------------------------
class DivergenceMetric(Protocol):
    """A divergence metric returns a non-negative scalar. 0 = identical."""

    name: str

    def __call__(self, reference: Sequence[float], candidate: Sequence[float]) -> float:
        ...


# ---------------------------------------------------------------------------
# Concrete metrics.
# ---------------------------------------------------------------------------
class CosineDistance:
    """
    1 - (a · b) / (|a| |b|).

    Range: [0, 2]. 0 = identical direction. 1 = orthogonal. 2 = opposite.

    Insensitive to magnitude. This is usually what you want for transformer
    activations because residual streams can have wildly different norms.
    Direction is what carries the information.
    """

    name = "cosine_distance"

    def __call__(self, reference: Sequence[float], candidate: Sequence[float]) -> float:
        if len(reference) != len(candidate):
            raise ValueError(
                f"cosine_distance: length mismatch "
                f"({len(reference)} vs {len(candidate)})"
            )
        if len(reference) == 0:
            return 0.0

        dot = 0.0
        ref_sq = 0.0
        cand_sq = 0.0
        for r, c in zip(reference, candidate):
            r_f = float(r)
            c_f = float(c)
            dot += r_f * c_f
            ref_sq += r_f * r_f
            cand_sq += c_f * c_f

        denom = math.sqrt(ref_sq) * math.sqrt(cand_sq)
        if denom == 0.0:
            # If either vector is all zeros, treat as identical only when both are.
            return 0.0 if (ref_sq == 0.0 and cand_sq == 0.0) else 1.0
        cosine_sim = dot / denom
        # Clip to [-1, 1] to handle floating-point drift.
        cosine_sim = max(-1.0, min(1.0, cosine_sim))
        return 1.0 - cosine_sim


class RelativeL2:
    """
    |a - b|_2 / |a|_2.

    Range: [0, +inf). 0 = identical. 1 = candidate is "noise" of similar
    magnitude as reference. >1 = candidate is further from reference than
    reference is from origin.

    Sensitive to magnitude. Use this when you care about absolute-error
    behavior, e.g. for attention output projections where scale matters.

    Falls back to absolute L2 if reference is zero (avoids divide-by-zero
    and matches the intuition that "candidate near zero is good when
    reference is zero").
    """

    name = "relative_l2"

    def __call__(self, reference: Sequence[float], candidate: Sequence[float]) -> float:
        if len(reference) != len(candidate):
            raise ValueError(
                f"relative_l2: length mismatch "
                f"({len(reference)} vs {len(candidate)})"
            )
        if len(reference) == 0:
            return 0.0

        diff_sq = 0.0
        ref_sq = 0.0
        for r, c in zip(reference, candidate):
            r_f = float(r)
            c_f = float(c)
            diff = r_f - c_f
            diff_sq += diff * diff
            ref_sq += r_f * r_f

        diff_norm = math.sqrt(diff_sq)
        ref_norm = math.sqrt(ref_sq)
        if ref_norm == 0.0:
            return diff_norm
        return diff_norm / ref_norm


# Singletons for the common case. Use these unless you need the class.
cosine_distance: DivergenceMetric = CosineDistance()
relative_l2: DivergenceMetric = RelativeL2()


# ---------------------------------------------------------------------------
# Metric registry — lookup by name (used by CLI argparse and schema).
# ---------------------------------------------------------------------------
_METRICS: dict[str, DivergenceMetric] = {
    cosine_distance.name: cosine_distance,
    relative_l2.name: relative_l2,
}


def get_metric(name: str) -> DivergenceMetric:
    if name not in _METRICS:
        raise ValueError(
            f"Unknown divergence metric: {name!r}. "
            f"Available: {sorted(_METRICS.keys())}"
        )
    return _METRICS[name]


def available_metrics() -> tuple[str, ...]:
    return tuple(sorted(_METRICS.keys()))


# ---------------------------------------------------------------------------
# Per-cell aggregation.
# ---------------------------------------------------------------------------
def aggregate_losses(losses: Iterable[float]) -> tuple[float, float, int, float, float, float]:
    """
    Aggregate raw per-sample losses into (mean, variance, n, min, max, median).

    Uses the unbiased sample variance estimator (n-1 denominator) when n>=2;
    returns 0 variance for n<2 to avoid NaNs.

    Median is computed by sorting; for n up to ~10k this is cheap. If the
    runner ever batches into the millions, swap for a streaming quantile.
    """
    values = sorted(losses)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0, 0.0, 0.0, 0.0

    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, 1, values[0], values[0], values[0]

    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    if n % 2 == 1:
        median = values[n // 2]
    else:
        median = (values[n // 2 - 1] + values[n // 2]) / 2.0
    return mean, var, n, values[0], values[-1], median
