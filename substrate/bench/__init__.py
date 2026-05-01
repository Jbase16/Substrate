"""
substrate.bench — benchmarking utilities.

Helpers for constructing test/benchmark plans, harnesses, and metric
computation. These are NOT part of the production runtime, but they
ARE first-class library code — benchmarks are the truth-tellers, not
afterthoughts.

Layout:

    plan_builders.py — build PlanBundles for tests + benchmarks without
                        going through the full RDO solver. Synthesizes
                        the minimum that the IR validator demands.

The previous home for these helpers was scripts/test_weight_bank.py.
That worked for tests-importing-tests but broke when the benchmark
harness tried to import them — `scripts/` has no `__init__.py` and
the dependency arrow pointed the wrong way (production code reaching
into test scaffolding). Moved here in 2026-05.
"""

from substrate.bench.plan_builders import (
    build_test_plan,
    build_op_tier_precisions,
)

__all__ = [
    "build_test_plan",
    "build_op_tier_precisions",
]
