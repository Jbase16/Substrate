"""
substrate.backend.ram_weight_store — In-memory packed quantized weights.

Builds a WeightStore by:
    1. Walking each op's target module (attention or mlp).
    2. For each tier in the plan, calling mx.quantize on every leaf weight
       to produce packed (quantized, scales, biases) tuples.
    3. Storing the packed forms in a dict, indexed by (op_id, tier).

The packed forms are MLX arrays held in RAM. For Qwen2.5-1.5B with three
tiers (3-bit, 4-bit, 6-bit), total bank cost is roughly:
    1.4B params × (3+4+6)/8 bytes = ~2.3 GB packed
plus the bias arrays (small) and scales (small).

The dequantized "active" arrays exist outside the store, in WeightBank.
The store holds packed; the bank holds the one currently-active dequant.

Construction is offline (a few seconds of mx.quantize calls). After that,
get() is a dict lookup.

============================================================================
QWEN2-SHAPED:
This file's _module_for_op() helper assumes Qwen2 layout: layer.self_attn
for attention, layer.mlp for mlp_dense / moe_dispatch. Same architectural
boundary as the kernel — the IR is generic, this isn't.
============================================================================
"""

from __future__ import annotations

import logging
from typing import Iterator, Iterable

from substrate.backend.weight_store import (
    PackedLeaf, PackedWeight, WeightStore,
)
from substrate.backend.quantize import (
    GROUP_SIZE, snapshot_weights, validate_bits,
)

log = logging.getLogger(__name__)


# Map planner Precision values <-> bit counts.
# NEAR_FP16 is special: bits=None signals "no quantization, use FP16 ref."
_PRECISION_TO_BITS: dict[str, int | None] = {
    "2bit": 2,
    "3bit": 3,
    "4bit": 4,
    "6bit": 6,
    "fp16_eq": None,
}


# ---------------------------------------------------------------------------
# RAMWeightStore.
# ---------------------------------------------------------------------------
class RAMWeightStore(WeightStore):
    """
    Holds packed quantized tensors in MLX arrays. Built once from a plan
    + a live MLX model.

    Construction parameters:
        model: the FP16 mlx-lm-loaded model whose weights we'll quantize.
        layers: the model's transformer block list.
        ops_with_tiers: for each op_id, the set of tier indices to materialize
                        (typically the tier indices admitted by the compiled
                        plan: tier 0 + tier 1 + tier 2 if present).
        op_tier_precisions: mapping (op_id, tier_index) -> Precision string
                            (the planner's value, e.g. '2bit', '4bit',
                            'fp16_eq').

    The store does NOT copy the model's FP16 weights into a separate FP16
    bank entry per op. The "FP16 reference" tier reuses pointers to the
    live module's original arrays, captured at store construction time
    BEFORE any other quantization gets installed. This means:
        - Memory cost of NEAR_FP16 entries is zero (just references).
        - The store can hand back the originals exactly via PackedWeight
          when the bank swaps to NEAR_FP16.
    """

    def __init__(
        self,
        model: object,
        layers: list[object],
        ops_with_tiers: dict[str, list[int]],
        op_tier_precisions: dict[tuple[str, int], str],
        *,
        group_size: int = GROUP_SIZE,
    ) -> None:
        self._model = model
        self._layers = layers
        self._group_size = group_size
        self._packed: dict[tuple[str, int], PackedWeight] = {}

        log.info(
            "RAMWeightStore: building bank for %d ops × variable tiers (group_size=%d)",
            len(ops_with_tiers), group_size,
        )
        self._build(ops_with_tiers, op_tier_precisions)
        log.info(
            "RAMWeightStore ready: %d (op, tier) entries", len(self._packed),
        )

    # ------------------------------------------------------------------
    # WeightStore protocol.
    # ------------------------------------------------------------------
    def get(self, op_id: str, tier: int) -> PackedWeight:
        try:
            return self._packed[(op_id, tier)]
        except KeyError:
            raise KeyError(
                f"RAMWeightStore: no entry for ({op_id!r}, tier {tier}). "
                f"Either the plan references a tier outside what the store "
                f"was built for, or the store wasn't built against this plan."
            )

    def has(self, op_id: str, tier: int) -> bool:
        return (op_id, tier) in self._packed

    def known_ops(self) -> Iterator[str]:
        seen: set[str] = set()
        for op_id, _ in self._packed:
            if op_id not in seen:
                seen.add(op_id)
                yield op_id

    def known_tiers(self, op_id: str) -> Iterator[int]:
        for stored_op, tier in self._packed:
            if stored_op == op_id:
                yield tier

    # ------------------------------------------------------------------
    # Construction.
    # ------------------------------------------------------------------
    def _build(
        self,
        ops_with_tiers: dict[str, list[int]],
        op_tier_precisions: dict[tuple[str, int], str],
    ) -> None:
        for op_id, tiers in ops_with_tiers.items():
            target_module = self._module_for_op(op_id)
            if target_module is None:
                raise RuntimeError(
                    f"RAMWeightStore: cannot resolve module for op {op_id!r}. "
                    f"Verify the model is Qwen2-shaped."
                )
            # Capture the FP16 originals BEFORE we quantize anything. The
            # bank will use these for any NEAR_FP16 tier entries.
            fp16_snapshots = self._snapshot_with_paths(target_module)

            for tier in tiers:
                precision = op_tier_precisions.get((op_id, tier))
                if precision is None:
                    raise KeyError(
                        f"op_tier_precisions missing entry for ({op_id!r}, tier {tier})"
                    )
                bits = _PRECISION_TO_BITS.get(precision)
                if precision != "fp16_eq" and bits is None:
                    raise ValueError(
                        f"Unknown precision string: {precision!r}"
                    )

                if bits is None:
                    # NEAR_FP16: reference the originals.
                    fp16_leaves = tuple(
                        (path, attr, arr) for path, attr, arr in fp16_snapshots
                    )
                    self._packed[(op_id, tier)] = PackedWeight(
                        op_id=op_id,
                        tier=tier,
                        bits=None,
                        leaves=(),
                        fp16_leaves=fp16_leaves,
                    )
                else:
                    # Quantize each eligible leaf.
                    leaves = self._quantize_snapshots(fp16_snapshots, bits)
                    self._packed[(op_id, tier)] = PackedWeight(
                        op_id=op_id,
                        tier=tier,
                        bits=bits,
                        leaves=leaves,
                    )

    def _quantize_snapshots(
        self,
        fp16_snapshots: list[tuple[tuple[str, ...], str, object]],
        bits: int,
    ) -> tuple[PackedLeaf, ...]:
        import mlx.core as mx
        validate_bits(bits)

        out: list[PackedLeaf] = []
        for path, attr, original in fp16_snapshots:
            # Eligibility: matches the rule used by quantize_module_in_place
            # and the calibration backend. mx.quantize requires 2D+ arrays
            # with last dim >= group_size.
            if original.ndim < 2 or min(original.shape) < self._group_size:
                continue
            q, scales, biases = mx.quantize(
                original, group_size=self._group_size, bits=bits,
            )
            # Force eval so the packed buffers are materialized now, not
            # lazily during the swap path. Eager construction = predictable
            # memory behavior.
            mx.eval(q, scales, biases)
            out.append(PackedLeaf(
                parent_path=path,
                attr_name=attr,
                quantized=q,
                scales=scales,
                biases=biases,
                bits=bits,
                group_size=self._group_size,
            ))
        return tuple(out)

    # ------------------------------------------------------------------
    # Module + path helpers.
    # ------------------------------------------------------------------
    def _module_for_op(self, op_id: str) -> object | None:
        """
        Resolve op_id to a module reference. Op IDs follow Substrate's
        convention: 'layer_{N}.{op_kind}'.

        Qwen2-shaped: attention -> layer.self_attn, mlp_dense -> layer.mlp.
        """
        try:
            layer_part, kind = op_id.split(".", 1)
            layer_id = int(layer_part[len("layer_"):])
        except (ValueError, IndexError):
            raise ValueError(f"Malformed op_id: {op_id!r}")

        if not (0 <= layer_id < len(self._layers)):
            raise IndexError(
                f"op {op_id}: layer_id {layer_id} out of range "
                f"({len(self._layers)} layers)"
            )
        layer = self._layers[layer_id]

        if kind == "attention":
            return getattr(layer, "self_attn", None) \
                or getattr(layer, "attention", None) \
                or getattr(layer, "attn", None)
        if kind in ("mlp_dense", "moe_dispatch"):
            return getattr(layer, "mlp", None) \
                or getattr(layer, "feed_forward", None) \
                or getattr(layer, "ffn", None)
        return None

    def _snapshot_with_paths(
        self, target_module: object,
    ) -> list[tuple[tuple[str, ...], str, object]]:
        """
        Walk the target module, returning (path_to_parent, attr_name,
        original_array) tuples for every weight leaf.

        path_to_parent is the dotted attribute path FROM target_module
        TO the leaf's parent. E.g. for attention's q_proj.weight:
            path = ('q_proj',)
            attr = 'weight'
        For target_module itself holding a leaf:
            path = ()
            attr = '<leaf-name>'

        We need the path because the bank's swap path has to walk it again
        to install the dequantized array back in the right place. We can't
        keep a direct parent reference because the parent submodule's
        identity is stable across swaps but our 'live' arrays are not.
        """
        out: list[tuple[tuple[str, ...], str, object]] = []
        self._walk(target_module, (), out)
        return out

    def _walk(
        self,
        module: object,
        path: tuple[str, ...],
        out: list[tuple[tuple[str, ...], str, object]],
    ) -> None:
        try:
            items = (
                module.items()
                if hasattr(module, "items") and callable(module.items)
                else []
            )
        except Exception:
            items = []

        for attr, value in items:
            if hasattr(value, "shape") and hasattr(value, "dtype") and not callable(value):
                out.append((path, attr, value))
            elif hasattr(value, "items") and callable(getattr(value, "items", None)):
                self._walk(value, path + (attr,), out)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if hasattr(item, "items") and callable(getattr(item, "items", None)):
                        # MoE-style indexed children. We encode the index
                        # as part of the path. The bank installer will
                        # handle this via getattr-then-getitem.
                        self._walk(item, path + (attr, str(i)), out)
