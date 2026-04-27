"""
substrate.calibration.synthetic_backend — Test backend with no MLX dependency.

Generates plausible activations from a deterministic RNG and produces synthetic
divergence patterns: lower precision -> higher loss, sensitive layers -> more
loss at low precision. Used to unit-test the calibration runner without
loading a real model.

The synthetic backend is also a useful smoke test for the schema: its output
has the same shape as the MLX backend's, so anything downstream that consumes
calibration data can be tested end-to-end without GPU access.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Sequence

from substrate.calibration.backend import (
    ActivationCapture,
    CalibrationBackend,
    OpAblation,
    OpDescriptor,
)


# ---------------------------------------------------------------------------
# A trivial activation capture: just a list of floats.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _ListCapture:
    values: tuple[float, ...]
    _shape: tuple[int, ...]

    def flatten(self) -> Sequence[float]:
        return self.values

    def shape(self) -> tuple[int, ...]:
        return self._shape


# ---------------------------------------------------------------------------
# Synthetic backend.
# ---------------------------------------------------------------------------
class SyntheticBackend:
    """
    Pretends to be an MLX-LM model. Useful for testing.

    The model is parameterized by num_layers, hidden_size, op_kinds_per_layer.
    Activation generation is deterministic per (sequence, op_id) so reference
    captures are stable across calls.

    Ablation behavior simulates monotonic quality decay: lower bits = larger
    cosine distance from the FP reference, with a sensitivity factor per layer
    that emphasizes drift in user-specified "sensitive" layers.
    """

    def __init__(
        self,
        model_id: str = "synthetic-test-model",
        num_layers: int = 4,
        hidden_size: int = 64,
        op_kinds_per_layer: tuple[str, ...] = ("attention", "mlp_dense"),
        sensitive_layers: tuple[int, ...] = (),
        seed: int = 0,
    ) -> None:
        self._model_id = model_id
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._op_kinds = op_kinds_per_layer
        self._sensitive = set(sensitive_layers)
        self._seed = seed

    @property
    def name(self) -> str:
        return "synthetic"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def discover_ops(self) -> tuple[OpDescriptor, ...]:
        ops = []
        for layer in range(self._num_layers):
            for kind in self._op_kinds:
                # Synthetic param counts: attention smaller than mlp.
                params = (
                    4 * self._hidden_size * self._hidden_size
                    if kind == "attention"
                    else 3 * self._hidden_size * (4 * self._hidden_size)
                )
                ops.append(OpDescriptor(
                    op_id=f"layer_{layer}.{kind}",
                    layer_id=layer,
                    op_kind=kind,
                    param_count=params,
                ))
        return tuple(ops)

    def encode_corpus(
        self, text: str, max_sequences: int, sequence_length: int,
    ) -> tuple[Sequence[int], ...]:
        # Synthetic tokenization: hash chunks of text into pseudo-token-ids.
        # Real tokenizers are sub-word; we just emit one int per character
        # for testing. Good enough to drive the runner's loop.
        sequences = []
        cursor = 0
        while cursor < len(text) and len(sequences) < max_sequences:
            chunk = text[cursor:cursor + sequence_length]
            tokens = tuple(ord(c) % 32000 for c in chunk)
            if not tokens:
                break
            sequences.append(tokens)
            cursor += sequence_length
        return tuple(sequences)

    def capture_reference(
        self, sequence: Sequence[int],
    ) -> dict[str, ActivationCapture]:
        """
        Generate a deterministic per-op activation. The seed is derived from
        the sequence content so the same prompt produces the same activation.
        """
        seq_seed = self._seed_from_sequence(sequence)
        captures: dict[str, ActivationCapture] = {}
        for op in self.discover_ops():
            rng = random.Random(seq_seed ^ hash(op.op_id))
            values = tuple(rng.gauss(0.0, 1.0) for _ in range(self._hidden_size))
            captures[op.op_id] = _ListCapture(
                values=values, _shape=(self._hidden_size,),
            )
        return captures

    def ablate_op(
        self, sequence: Sequence[int], ablation: OpAblation,
    ) -> ActivationCapture:
        """
        Generate an ablated activation that diverges from the FP reference.

        Divergence increases as bits decrease. Sensitive layers diverge more
        at the same bit-width. The relationship is: noise std = base_noise *
        precision_factor * sensitivity_factor.
        """
        seq_seed = self._seed_from_sequence(sequence)
        rng = random.Random(seq_seed ^ hash(ablation.op_id))
        # Reproduce the FP reference deterministically...
        ref_values = [rng.gauss(0.0, 1.0) for _ in range(self._hidden_size)]

        # ...then perturb based on ablation params.
        precision_factor = {2: 0.40, 3: 0.20, 4: 0.10, 6: 0.04, 8: 0.02}.get(
            ablation.precision_bits, 0.10,
        )
        sensitivity_factor = 2.5 if ablation.layer_id in self._sensitive else 1.0
        noise_std = 1.0 * precision_factor * sensitivity_factor

        # Use an independent RNG for noise so it's seed-stable but not coupled
        # to the FP reference draw.
        noise_rng = random.Random(
            seq_seed ^ hash((ablation.op_id, ablation.precision_bits)),
        )
        noisy = tuple(
            v + noise_rng.gauss(0.0, noise_std) for v in ref_values
        )
        return _ListCapture(values=noisy, _shape=(self._hidden_size,))

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------
    def _seed_from_sequence(self, sequence: Sequence[int]) -> int:
        # Stable hash of the token sequence. Plain hash() is salted in modern
        # Python; we use a deterministic digest instead.
        h = hashlib.sha1()
        h.update(self._seed.to_bytes(8, "little", signed=False))
        for tok in sequence:
            h.update(int(tok).to_bytes(4, "little", signed=False))
        return int.from_bytes(h.digest()[:8], "little", signed=False)
