"""
substrate.backend.quantize — Shared MLX weight quantization helpers.

Hoisted from substrate.calibration.mlxlm_backend so calibration and the
runtime kernel use one quantization implementation. Calibration and
runtime using different quantization logic would silently produce
inconsistent quality estimates: the planner would say "this op at 4-bit
loses 0.001 quality" but the runtime would actually run a 4-bit op
with different rounding and different loss. That's a fatal class of bug;
one helper avoids it.

The two primitives:

    snapshot_weights(module) -> list of (parent, attr, original_array)
        Walks an mlx.nn.Module subtree, returns stable references to
        every weight array along with the parent and attribute name
        needed to restore them.

    temporarily_quantized(target_module, bits) — context manager that
        applies mx.quantize(W, group_size=64, bits=N) followed by
        mx.dequantize for every eligible weight in the subtree. On exit,
        ALWAYS restores originals — even on exception.

Eligibility rule: array.ndim >= 2 and min(array.shape) >= 64. mx.quantize
requires last-dim >= group_size (64); 1D biases and small tensors are
left untouched. This matches what mx.quantize_module does internally
when applied via mlx-lm's quantization paths.

Group size of 64 is mlx's default. We hardcode it because:
    - mlx-lm's `convert.py` uses 64 throughout.
    - The HF-published mlx-community quantized snapshots use 64.
    - Matching this means our calibration measures the same numerical
      behavior as the published quantized models.

If we ever need a different group size, plumbing it through means
threading a parameter; for now, single-knob simplicity is the right
tradeoff.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

log = logging.getLogger(__name__)


# Hardcoded mlx default. See module docstring for why we don't parameterize this.
GROUP_SIZE = 64

# Bits supported by mx.quantize.
SUPPORTED_BITS: tuple[int, ...] = (2, 3, 4, 6, 8)


def validate_bits(bits: int) -> None:
    if bits not in SUPPORTED_BITS:
        raise ValueError(
            f"Unsupported quantization bits: {bits}. "
            f"Supported by mx.quantize: {SUPPORTED_BITS}"
        )


# ---------------------------------------------------------------------------
# Weight tree walker.
# ---------------------------------------------------------------------------
def snapshot_weights(
    module: object,
    out: list[tuple[object, str, object]] | None = None,
) -> list[tuple[object, str, object]]:
    """
    Walk an mlx.nn.Module subtree and collect every weight array.

    Returns a list of (parent_module, attribute_name, original_array)
    triples. The triples can be used to restore originals via
    `setattr(parent, attr, original)` — that's how the temporarily_quantized
    context manager rolls back.

    mlx.nn.Module extends dict; we iterate via .items() to get
    (name, child) pairs. Leaf arrays are detected by having .shape and
    .dtype but not .items (i.e. they aren't submodules).

    Lists/tuples are recursed into for MoE-style architectures where
    `experts` is a list of submodules.
    """
    if out is None:
        out = []
    try:
        items = module.items() if hasattr(module, "items") and callable(module.items) else []
    except Exception:
        items = []

    for attr, value in items:
        if hasattr(value, "shape") and hasattr(value, "dtype") and not callable(value):
            # Leaf array (weight or bias).
            out.append((module, attr, value))
        elif hasattr(value, "items") and callable(getattr(value, "items", None)):
            # Submodule. Recurse.
            snapshot_weights(value, out)
        elif isinstance(value, (list, tuple)):
            # List of submodules (e.g. MoE experts).
            for item in value:
                if hasattr(item, "items") and callable(getattr(item, "items", None)):
                    snapshot_weights(item, out)
    return out


# ---------------------------------------------------------------------------
# Context manager: quantize subtree, restore on exit.
# ---------------------------------------------------------------------------
@contextmanager
def temporarily_quantized(
    target_module: object,
    bits: int,
    *,
    group_size: int = GROUP_SIZE,
) -> Iterator[None]:
    """
    Quantize-then-dequantize every eligible weight in `target_module`'s
    subtree, run the wrapped code, then restore originals.

    Why round-trip instead of just quantize?
        mx.quantize produces a packed integer array plus scales/biases.
        Running the model with that packed form requires mx.quantized_matmul
        kernels — which only work in specific module types (QuantizedLinear).
        The round-trip dequantize gives us a normal floating-point array
        with the same numerical content as the lower-precision form, so
        it's a drop-in replacement for the original weights.
        Real production runtime will use packed integer storage + quantized
        kernels for memory savings; calibration and Test 1 use the
        round-trip for simplicity and equivalence.

    Eligibility: array.ndim >= 2 and min(shape) >= group_size. Skips 1D
    biases and small tensors (which mx.quantize would reject anyway).

    Always restores. The finally block runs even if the wrapped code
    raises, so a failed forward pass doesn't poison subsequent calls.
    """
    import mlx.core as mx
    validate_bits(bits)

    snapshots: list[tuple[object, str, object]] = []
    snapshot_weights(target_module, snapshots)

    if not snapshots:
        log.warning("temporarily_quantized: no weight arrays found in module")

    try:
        for parent, attr, original in snapshots:
            if original.ndim < 2 or min(original.shape) < group_size:
                continue
            quantized, scales, biases = mx.quantize(
                original, group_size=group_size, bits=bits,
            )
            dequantized = mx.dequantize(
                quantized, scales, biases, group_size=group_size, bits=bits,
            )
            setattr(parent, attr, dequantized)
        yield
    finally:
        for parent, attr, original in snapshots:
            setattr(parent, attr, original)


# ---------------------------------------------------------------------------
# Bulk quantization. Used by Test 1 (uniform 4-bit applied across the model
# all at once, not in a context manager).
# ---------------------------------------------------------------------------
def quantize_module_in_place(
    target_module: object,
    bits: int,
    *,
    group_size: int = GROUP_SIZE,
) -> int:
    """
    Apply quantize-then-dequantize round-trip to every eligible weight in
    target_module's subtree. Mutates the module in place. Returns the
    number of weight arrays modified.

    NOT reversible. Use this when you want to permanently set a model's
    weights to lower-precision-equivalent values. For reversible scoped
    quantization, use temporarily_quantized.
    """
    import mlx.core as mx
    validate_bits(bits)

    snapshots: list[tuple[object, str, object]] = []
    snapshot_weights(target_module, snapshots)

    modified = 0
    for parent, attr, original in snapshots:
        if original.ndim < 2 or min(original.shape) < group_size:
            continue
        quantized, scales, biases = mx.quantize(
            original, group_size=group_size, bits=bits,
        )
        dequantized = mx.dequantize(
            quantized, scales, biases, group_size=group_size, bits=bits,
        )
        setattr(parent, attr, dequantized)
        modified += 1
    return modified
