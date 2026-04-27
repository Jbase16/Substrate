"""
substrate.calibration.quantize — Weight swap context manager for ablation.

The calibration runner needs to ablate one op at a time: temporarily quantize
its weights, run a forward pass, restore the originals. This must be:

    1. Atomic — restoration happens even if the forward pass raises.
    2. Local  — only the targeted op is affected; everything else stays FP.
    3. Cheap  — quantize/dequantize round-trip should be measured in ms,
                not seconds, because we do this thousands of times per run.

Usage:

    with quantized_module_weights(model, "model.layers.5.self_attn", bits=4):
        output = model(tokens)
    # weights are restored here, even if model(tokens) raised

The implementation lives in mlxlm_backend because it requires MLX. This file
defines the protocol so other backends (PyTorch, future) can implement the
same contract without depending on MLX.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Protocol


class WeightSwapper(Protocol):
    """
    A backend-specific helper that knows how to swap a module's weights.

    The protocol is exposed so the runner can pass the swapper into the
    backend without the runner itself importing MLX.
    """

    @contextmanager
    def temporarily_quantized(
        self, module_path: str, bits: int,
    ) -> Iterator[None]:
        """
        Context manager that quantizes the module's weights to `bits` on entry,
        restores them on exit (always — even on exception).

        Implementations must:
            - Save the original weights before quantizing.
            - Apply the quantize-then-dequantize round-trip in place, so the
              forward pass sees lower-precision-equivalent weights.
            - Restore the originals in a `finally` block.

        bits must be one of {2, 3, 4, 6}. mlx.quantize does not support 5 or 7.
        Callers should validate before calling.
        """
        ...


# ---------------------------------------------------------------------------
# Validation helper.
# ---------------------------------------------------------------------------
SUPPORTED_BITS: tuple[int, ...] = (2, 3, 4, 6, 8)


def validate_bits(bits: int) -> None:
    if bits not in SUPPORTED_BITS:
        raise ValueError(
            f"Unsupported quantization bits: {bits}. "
            f"Supported: {SUPPORTED_BITS}"
        )
