"""
substrate.backend.weight_store — Storage abstraction for quantized weights.

Defines the seam between "where do quantized weight bytes live" and "how
does the runtime use them." The WeightBank depends on this protocol, not
on any specific backend.

v0.1: only RAMWeightStore is implemented — packed quantized tensors held
in MLX arrays, in process memory.

v1.x (not yet): MMapWeightStore reads packed weights from a QuantizedStore
on disk via mmap. Same protocol; WeightBank doesn't change.

Storage contract:

    A "packed weight" for one (op_id, tier) is a small object that knows
    how to materialize itself into the FP16-equivalent dequantized arrays
    needed by the live MLX module. The packed form is whatever the
    backend chose: tuples of (quantized_int_array, scales, biases) in v0.1,
    file offsets in a future SSD-backed implementation.

Why a separate object instead of returning arrays directly:

    Packed weights from mlx.quantize are 3-tuples (q, scales, biases),
    one per leaf tensor in the module. A module has multiple leaf tensors
    (q_proj, k_proj, v_proj, o_proj for attention). The PackedWeight bundles
    them and knows how to dequantize the set as a unit.

    Going from packed -> dequantized is the only thing the runtime calls
    in the swap path. Everything else (where the bytes came from, how
    they're stored, whether they're memory-mapped) is the WeightStore's
    business.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Mapping, Protocol

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Packed weight: backend-agnostic carrier.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PackedLeaf:
    """
    One leaf tensor's packed form, plus the path needed to install it back.

    parent_path: the dotted attribute path INSIDE the op's target module
                 needed to reach this leaf. E.g. for attention, this might
                 be ('q_proj',) — meaning the leaf is at module.q_proj.weight.
    attr_name:   the attribute on the parent that holds the array (usually
                 'weight'; could be other names for unusual modules).
    quantized:   the packed integer array from mx.quantize.
    scales:      per-group scale factors.
    biases:      per-group bias offsets.
    bits:        the precision used to pack.
    group_size:  the group size used to pack.
    """
    parent_path: tuple[str, ...]
    attr_name: str
    quantized: object   # mx.array; opaque to keep this file MLX-import-free
    scales: object
    biases: object
    bits: int
    group_size: int


@dataclass(frozen=True)
class PackedWeight:
    """
    The packed form for ONE op at ONE tier. May contain multiple leaves
    (e.g. four leaves for attention: q/k/v/o projections).

    For tier=NEAR_FP16 (no quantization), `leaves` is empty and `fp16_leaves`
    holds the original FP16 arrays. The dequantize() method handles both
    cases uniformly.
    """
    op_id: str
    tier: int
    bits: int | None    # None => NEAR_FP16 reference, no quantization applied
    leaves: tuple[PackedLeaf, ...]
    # FP16 reference path: when bits is None, leaves is empty and we hold
    # direct references to the original arrays. Storing them here means the
    # bank can install them without special-casing the FP16 swap path.
    fp16_leaves: tuple[tuple[tuple[str, ...], str, object], ...] = ()

    def dequantize(self) -> "DequantizedWeight":
        """
        Materialize the FP16-equivalent dequantized arrays.

        For quantized tiers: runs mx.dequantize once per leaf. The result
        is a fresh FP16 array per leaf; the caller is expected to install
        them and free any prior active arrays.

        For NEAR_FP16: returns references to the original FP16 arrays
        (no copies made). This is correct because:
            - The originals never get mutated by the bank itself.
            - Reusing the references is what makes "swap to FP16" cheap.
        """
        import mlx.core as mx

        if self.bits is None:
            # NEAR_FP16 path. Just hand back the original arrays.
            return DequantizedWeight(
                op_id=self.op_id,
                tier=self.tier,
                installations=tuple(
                    _Install(parent_path=path, attr_name=attr, array=arr)
                    for path, attr, arr in self.fp16_leaves
                ),
            )

        installations = []
        for leaf in self.leaves:
            arr = mx.dequantize(
                leaf.quantized, leaf.scales, leaf.biases,
                group_size=leaf.group_size, bits=leaf.bits,
            )
            installations.append(_Install(
                parent_path=leaf.parent_path,
                attr_name=leaf.attr_name,
                array=arr,
            ))
        return DequantizedWeight(
            op_id=self.op_id,
            tier=self.tier,
            installations=tuple(installations),
        )


@dataclass(frozen=True)
class _Install:
    """
    One leaf array ready to install on the live module.
    Internal to the dequantize() return type.
    """
    parent_path: tuple[str, ...]
    attr_name: str
    array: object   # mx.array


@dataclass(frozen=True)
class DequantizedWeight:
    """
    Fully materialized FP16 arrays for one op at one tier, with the paths
    needed to install them back on the live module.

    The bank consumes one of these per swap, walks the installations,
    and replaces the live module's weight pointers.
    """
    op_id: str
    tier: int
    installations: tuple[_Install, ...]


# ---------------------------------------------------------------------------
# WeightStore protocol.
# ---------------------------------------------------------------------------
class WeightStore(Protocol):
    """
    Source of packed weights. The bank knows nothing about where bytes
    physically live — only that the store can produce a PackedWeight
    given an (op_id, tier) pair.

    Implementations must be deterministic: the same (op_id, tier) returns
    PackedWeights that dequantize to byte-identical arrays. This matters
    for testing — if the store is non-deterministic, swap loops will
    produce drifting numerics that hide real bugs.

    Implementations need not be thread-safe; the bank serializes access.
    """

    def get(self, op_id: str, tier: int) -> PackedWeight:
        """Return the packed weights for op_id at the given tier."""
        ...

    def has(self, op_id: str, tier: int) -> bool:
        """Cheap existence check; used to validate plans against the store."""
        ...

    def known_ops(self) -> Iterator[str]:
        """Enumerate op_ids the store has data for."""
        ...

    def known_tiers(self, op_id: str) -> Iterator[int]:
        """Enumerate tiers available for op_id."""
        ...
