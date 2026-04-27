"""
substrate.calibration.backend — Pluggable model backend protocol.

The CalibrationBackend protocol is the seam between the calibration runner
and the actual model execution layer. Two implementations ship with v0.1:

    MLXLMBackend   — Real path. Loads an mlx-lm model, captures activations,
                     ablates ops via mx.quantize round-trip.
    SyntheticBackend — Test path. Generates plausible activation tensors
                       without loading a real model. Used for unit tests of
                       the runner orchestration logic.

Both implement the same three operations:

    1. discover_ops()      — Tell the runner what ops the model has.
    2. capture_reference() — Run FP forward pass, capture activations.
    3. ablate_op()         — Run with one op quantized, capture its output.

Activations are returned as ActivationCapture objects: opaque from the
runner's perspective. The metrics module accepts flat float sequences,
so ActivationCapture.flatten() is the bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, Sequence


# ---------------------------------------------------------------------------
# What the runner asks the backend to do.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpAblation:
    """
    A single ablation request: 'run the forward pass with `op_id` quantized
    to `precision_bits`, and give me the output activations of that op'.
    """
    op_id: str          # Backend-specific identifier (e.g. "model.layers.5.self_attn")
    layer_id: int       # Logical layer index (0..num_layers-1)
    op_kind: str        # Coarse taxonomy: attention | mlp_dense | moe_router | moe_dispatch
    precision_bits: int # Target bit-width: 2, 3, 4, 6


@dataclass(frozen=True)
class OpDescriptor:
    """
    What the backend reports about each ablatable op. The runner uses this
    to plan the ablation sweep without knowing the model architecture.
    """
    op_id: str
    layer_id: int
    op_kind: str
    param_count: int

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise ValueError(f"{self.op_id}: layer_id must be >= 0")
        if self.param_count < 0:
            raise ValueError(f"{self.op_id}: param_count must be >= 0")


# ---------------------------------------------------------------------------
# What the backend returns: activation captures.
# ---------------------------------------------------------------------------
class ActivationCapture(Protocol):
    """
    An opaque activation tensor handle. The runner doesn't care what's inside;
    it just calls flatten() to get a float sequence for the metric.

    Implementations may store the activation as MLX array, numpy array, or
    plain Python list. flatten() is required to be deterministic — calling it
    twice on the same capture must produce the same sequence in the same order.
    """

    def flatten(self) -> Sequence[float]:
        """Return the activation as a flat sequence of floats."""
        ...

    def shape(self) -> tuple[int, ...]:
        """Return the activation's logical shape, for diagnostics."""
        ...


# ---------------------------------------------------------------------------
# The backend protocol itself.
# ---------------------------------------------------------------------------
class CalibrationBackend(Protocol):
    """
    Interface every calibration backend must implement.

    Lifecycle:
        backend = SomeBackend(model_id, ...)
        ops = backend.discover_ops()
        for sequence in corpus:
            ref_activations = backend.capture_reference(sequence)
            for ablation in ablations_to_run:
                cand = backend.ablate_op(sequence, ablation)
                # runner computes metric(ref_activations[ablation.op_id], cand)
        backend.close()

    Implementations must be thread-safe across capture_reference and ablate_op
    only if the runner uses parallel ablation sweeps (it doesn't in v0.1).
    Single-threaded is fine.
    """

    @property
    def name(self) -> str:
        """Short identifier, e.g. 'mlx-lm' or 'synthetic'. Stored in RunConfig."""
        ...

    @property
    def version(self) -> str:
        """Backend library version, e.g. mlx-lm version. Stored in RunConfig."""
        ...

    @property
    def model_id(self) -> str:
        """Resolved model identifier, e.g. 'mlx-community/Qwen2.5-1.5B-Instruct-4bit'."""
        ...

    @property
    def num_layers(self) -> int:
        ...

    def discover_ops(self) -> tuple[OpDescriptor, ...]:
        """
        Enumerate every op the backend can ablate.

        Returns ops in topological order (layer 0 first). The runner expects
        that ops are grouped by layer for the coarse taxonomy: e.g. an
        attention layer might decompose into q_proj/k_proj/v_proj/o_proj at
        the model level, but the backend reports a single 'attention' op
        per layer with the aggregated param_count.
        """
        ...

    def encode_corpus(self, text: str, max_sequences: int, sequence_length: int) \
            -> tuple[Sequence[int], ...]:
        """
        Tokenize the corpus into a tuple of token-id sequences.

        Each sequence has at most `sequence_length` tokens. The corpus is
        chunked greedily — long texts produce multiple sequences. Total
        sequences is capped at `max_sequences`.

        Returning sequences (not strings) is intentional: the runner can
        log the corpus_sha256 of the original text while the backend works
        in token space.
        """
        ...

    def capture_reference(
        self, sequence: Sequence[int],
    ) -> dict[str, ActivationCapture]:
        """
        Run the FP forward pass on `sequence` and return per-op output
        activations. Keys are op_ids matching what discover_ops returned.

        For coarse taxonomy: an 'attention' op's activation is the layer's
        attn output (post-projection, pre-residual-add). An 'mlp_dense'
        op's activation is the layer's MLP output (post-down-projection,
        pre-residual-add). This is what we'll compare against ablations.
        """
        ...

    def ablate_op(
        self, sequence: Sequence[int], ablation: OpAblation,
    ) -> ActivationCapture:
        """
        Run the forward pass with `ablation.op_id` quantized to
        `ablation.precision_bits`, and return that op's output activation.

        The ablation MUST be reverted before this method returns — the
        backend is responsible for restoring original weights so subsequent
        calls see clean state. This is enforced via context manager pattern
        in the MLX backend.
        """
        ...

    def close(self) -> None:
        """Release any resources (model weights, GPU memory). Idempotent."""
        ...
