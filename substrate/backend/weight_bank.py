"""
substrate.backend.weight_bank — Per-op precision swapping with O(1) state.

The bank is the runtime primitive that makes tier escalation cheap. It
sits between the kernel and a WeightStore. The kernel asks the bank
"this op is at tier N for this token"; the bank either does nothing
(already there) or installs new dequantized weights from the store.

State the bank keeps:

    self._active_tier:     op_id -> int          # which tier is live now
    self._active_arrays:   op_id -> tuple[arrays] # the live FP16 arrays
                                                   # (kept so we can drop
                                                   # references on swap)

The bank does NOT keep state on the live MLX modules. There is no
'_substrate_active_tier' attribute being set on the layer's submodules.
The bank is the single source of truth.

Lifecycle:

    bank = WeightBank(model, layers, store, plan)
    bank.swap('layer_5.attention', target_tier=2)      # upgrade
    # ... forward pass uses the new weights ...
    bank.swap('layer_5.attention', target_tier=0)      # demote

    bank.reset_to_tier_0()    # bring everything back to plan default

Bank does not own the WeightStore — store is injected. v0.1 uses
RAMWeightStore. v1.x will plug in MMapWeightStore reading from a
QuantizedStore on disk; the bank itself doesn't change.

Memory behavior:

    On every swap, the previous active arrays for that op are released
    by clearing the bank's reference. MLX is lazily evaluated and may
    hold onto buffers internally; we explicitly call mx.eval on the new
    arrays AFTER installation to force materialization, which lets the
    Python-side reference drop happen at a known point.

    Tests should verify that repeated swaps don't grow process RSS.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from substrate.backend.weight_store import (
    DequantizedWeight, PackedWeight, WeightStore,
)

if TYPE_CHECKING:
    from substrate.compiler.ir import PlanBundle

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier-name resolution. The bank speaks tier_index (int); the planner
# speaks Precision (enum). RAMWeightStore was built with both maps available;
# the bank just needs the int -> int passthrough plus a way to find the
# 'install path' for each op.
# ---------------------------------------------------------------------------
class WeightBank:
    """
    Mediates between WeightStore (packed bytes) and live MLX modules
    (active dequantized arrays).

    Construction:
        bank = WeightBank(
            model=mlx_lm_model,
            layers=transformer_layers,
            store=ram_or_mmap_weight_store,
            plan=compiled_plan_bundle,
        )

    The bank initializes every op to its tier-0 weights (the plan's
    default execution path) on construction. After that, swap() is the
    only mutation interface.
    """

    def __init__(
        self,
        model: object,
        layers: list[object],
        store: WeightStore,
        plan: "PlanBundle",
    ) -> None:
        self._model = model
        self._layers = layers
        self._store = store
        self._plan = plan
        self._active_tier: dict[str, int] = {}
        # _active_arrays: op_id -> tuple of mx.array references.
        # Held so we can drop them on swap, ensuring MLX can free buffers.
        self._active_arrays: dict[str, tuple[object, ...]] = {}

        # Initialize every op to tier 0 (the plan's default).
        log.info("WeightBank: initializing %d ops to tier 0", plan.num_ops)
        for ob in plan.op_bundles:
            self.swap(ob.op_id, target_tier=0)
        log.info("WeightBank ready")

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------
    def active_tier(self, op_id: str) -> int:
        """The tier currently installed for op_id."""
        try:
            return self._active_tier[op_id]
        except KeyError:
            raise KeyError(
                f"op_id {op_id!r} not registered in this bank. "
                f"Either it's not in the plan or the bank wasn't initialized."
            )

    def swap(self, op_id: str, target_tier: int) -> None:
        """
        Install target_tier's weights for op_id. No-op if already there.

        The actual cost is one mx.dequantize per weight leaf in the op's
        module, plus the cost of running mx.eval to materialize the arrays.
        For Qwen2.5-1.5B attention (4 leaves) that's ~10ms total on M-series
        silicon. The pointer-installation step is microseconds.

        Frees the previous active arrays by dropping their refcount; MLX
        may not release the underlying buffers immediately if they're
        still referenced by an in-flight forward pass, but the bank no
        longer holds them after this returns.
        """
        if self._active_tier.get(op_id) == target_tier:
            return

        # Pull the packed form from the store.
        packed = self._store.get(op_id, target_tier)
        # Materialize.
        dequant = packed.dequantize()
        # Install on live modules.
        new_arrays = self._install(op_id, dequant)
        # Drop the previous active set. Important for memory hygiene —
        # without this drop, the bank would steadily accumulate arrays.
        self._active_arrays[op_id] = new_arrays
        self._active_tier[op_id] = target_tier

        # Force evaluation. mx is lazy; without this, the install just
        # records graph nodes and the actual materialization is deferred
        # to forward time. We want predictable memory behavior, which means
        # eager materialization at swap time.
        import mlx.core as mx
        if new_arrays:
            mx.eval(*new_arrays)

    def reset_to_tier_0(self) -> None:
        """Demote every op back to its plan-default tier."""
        for ob in self._plan.op_bundles:
            self.swap(ob.op_id, target_tier=0)

    def active_state(self) -> dict[str, int]:
        """Snapshot of all op tiers. For diagnostics and tests."""
        return dict(self._active_tier)

    # ------------------------------------------------------------------
    # Module installation.
    # ------------------------------------------------------------------
    def _install(
        self, op_id: str, dequant: DequantizedWeight,
    ) -> tuple[object, ...]:
        """
        Walk the dequantized arrays and assign them onto the live module
        tree. Returns the tuple of installed arrays so the bank can hold
        references for later release.
        """
        target_module = self._module_for_op(op_id)
        if target_module is None:
            raise RuntimeError(
                f"WeightBank: cannot resolve module for op {op_id!r}"
            )

        installed: list[object] = []
        for inst in dequant.installations:
            parent = self._walk_path(target_module, inst.parent_path)
            setattr(parent, inst.attr_name, inst.array)
            installed.append(inst.array)
        return tuple(installed)

    @staticmethod
    def _walk_path(module: object, path: tuple[str, ...]) -> object:
        """
        Walk a parent path (list of attribute names, possibly with
        integer-string components for indexed lists like MoE experts).
        """
        cur = module
        for step in path:
            if step.isdigit():
                # Indexed access: previous step was a list/tuple attribute.
                # cur is the list; step is the index.
                cur = cur[int(step)]
            else:
                cur = getattr(cur, step)
        return cur

    def _module_for_op(self, op_id: str) -> object | None:
        """Same Qwen2-shaped logic as RAMWeightStore."""
        try:
            layer_part, kind = op_id.split(".", 1)
            layer_id = int(layer_part[len("layer_"):])
        except (ValueError, IndexError):
            raise ValueError(f"Malformed op_id: {op_id!r}")

        if not (0 <= layer_id < len(self._layers)):
            raise IndexError(
                f"op {op_id}: layer_id {layer_id} out of range"
            )
        layer = self._layers[layer_id]

        if kind == "attention":
            return getattr(layer, "self_attn", None)
        if kind in ("mlp_dense", "moe_dispatch"):
            return getattr(layer, "mlp", None)
        return None
