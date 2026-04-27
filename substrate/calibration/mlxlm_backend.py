"""
substrate.calibration.mlxlm_backend — Real backend using mlx-lm.

Implements CalibrationBackend against an mlx-lm model. Captures per-op
activations via forward hooks, and ablates ops via mx.quantize round-trip
in a context manager that restores weights even on exception.

This module imports mlx and mlx_lm at module level. It is only loaded when
the user actually invokes calibration; the calibration package's __init__
does NOT import it. Substrate's planner-only path stays MLX-free.

Architectural notes:

    Op discovery walks the model tree and identifies attention / mlp blocks
    by class name. We collapse fine-grained sub-modules (q_proj, k_proj, etc.)
    into single coarse ops ('attention', 'mlp_dense') matching Substrate's
    OpKind taxonomy. The ablation context manager quantizes ALL sub-modules
    of the coarse op simultaneously, so 'ablate attention at 4 bits' means
    'quantize q_proj, k_proj, v_proj, o_proj all at 4 bits'.

    Activation capture is via mlx-lm's hidden_states output if available,
    otherwise via direct module hooks. v0.1 uses hidden_states for the
    residual stream after each block, which is what matters for downstream
    behavior. Per-sub-module captures (e.g. attn output before residual)
    are V2 — they require deeper instrumentation.

    Because we use the residual-stream output as the "activation" for both
    'attention' and 'mlp_dense' in the same layer, divergence will overlap
    somewhat between those two ops. This is acceptable for v0.1: the
    estimator only needs relative ordering across precisions, not absolute
    isolation per sub-op. V2 can split them.
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

from substrate.calibration.backend import (
    ActivationCapture,
    OpAblation,
    OpDescriptor,
)
from substrate.calibration.quantize import validate_bits

log = logging.getLogger(__name__)


# Lazy imports: don't fail at import time if MLX isn't installed.
# The backend is only instantiated when the user invokes calibration with
# --backend mlx-lm, so an ImportError here is correct.
def _import_mlx():
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx_lm
        from mlx_lm import load as mlxlm_load
    except ImportError as e:
        raise ImportError(
            "MLXLMBackend requires the 'calibration' optional extra. "
            "Install with: pip install 'substrate[calibration]'"
        ) from e
    return mx, nn, mlx_lm, mlxlm_load


# ---------------------------------------------------------------------------
# Activation capture: a thin wrapper over an MLX array.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _MLXCapture:
    """
    Wraps an mlx.core.array. flatten() converts to Python floats lazily.

    We store the array, not the flattened sequence, because the runner often
    wants to compute multiple metrics over the same capture. Flattening on
    every metric call would be wasteful.
    """
    array: object               # mx.array; typed as object to avoid MLX import
    _shape_tuple: tuple[int, ...]

    def flatten(self) -> Sequence[float]:
        # Convert MLX array -> Python list of floats via numpy.
        # mx.array supports .tolist() in recent mlx-lm versions; fall back
        # to numpy conversion otherwise.
        try:
            flat = self.array.flatten().tolist()
            return tuple(float(v) for v in flat)
        except AttributeError:
            import numpy as np
            return tuple(float(v) for v in np.array(self.array).flatten())

    def shape(self) -> tuple[int, ...]:
        return self._shape_tuple


# ---------------------------------------------------------------------------
# The backend.
# ---------------------------------------------------------------------------
class MLXLMBackend:
    """
    Calibration backend backed by mlx-lm.

    Construct with a HuggingFace model id (e.g. 'mlx-community/Qwen2.5-1.5B-Instruct').
    Models that are already quantized in the snapshot WILL NOT be useful for
    calibration — we need FP16/BF16 weights to ablate from. Use the FP variant
    of any quantized model on the Hub.
    """

    def __init__(self, model_id: str) -> None:
        self._mx, self._nn, self._mlx_lm, self._load = _import_mlx()
        self._model_id_str = model_id

        log.info("Loading model %s via mlx-lm...", model_id)
        # mlx_lm.load returns (model, tokenizer). For some versions it returns
        # a config dict as well; handle both shapes.
        loaded = self._load(model_id)
        if len(loaded) == 2:
            self._model, self._tokenizer = loaded
        else:
            self._model, self._tokenizer = loaded[0], loaded[1]

        self._layers = self._discover_layers()
        log.info("Loaded %s with %d layers", model_id, len(self._layers))

    @property
    def name(self) -> str:
        return "mlx-lm"

    @property
    def version(self) -> str:
        return getattr(self._mlx_lm, "__version__", "unknown")

    @property
    def model_id(self) -> str:
        return self._model_id_str

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    # ------------------------------------------------------------------
    # Op discovery.
    # ------------------------------------------------------------------
    def _discover_layers(self) -> list[object]:
        """
        Walk the model tree and find the transformer block list.

        Different mlx-lm models put the layers in different attributes:
            - Llama/Qwen: model.model.layers
            - Some MoE models: model.model.layers (same, but with MoE inside)

        We try a list of common paths and use the first one that resolves.
        """
        candidates = [
            ("model", "layers"),                      # Most mlx-lm models
            ("model", "model", "layers"),             # Some wrappers
            ("transformer", "h"),                     # GPT-2-style
            ("layers",),                              # Bare model
        ]
        for path in candidates:
            obj = self._model
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok and isinstance(obj, (list, tuple)) and len(obj) > 0:
                return list(obj)
        raise RuntimeError(
            f"Could not locate transformer layer list in model {self._model_id_str}. "
            f"Tried: {candidates}. Inspect model structure and update _discover_layers."
        )

    def discover_ops(self) -> tuple[OpDescriptor, ...]:
        """
        For each layer, identify the attention block and the MLP block,
        then report one OpDescriptor each.

        Param counts are summed across all sub-modules of the coarse op.
        We use mlx's parameter tree for this.
        """
        ops: list[OpDescriptor] = []
        for layer_id, layer in enumerate(self._layers):
            attn = self._find_submodule(layer, ("self_attn", "attention", "attn"))
            mlp = self._find_submodule(layer, ("mlp", "feed_forward", "ffn"))

            if attn is not None:
                ops.append(OpDescriptor(
                    op_id=f"layer_{layer_id}.attention",
                    layer_id=layer_id,
                    op_kind="attention",
                    param_count=self._count_params(attn),
                ))
            if mlp is not None:
                # Detect MoE vs dense by structure: MoE has an experts list.
                if self._is_moe(mlp):
                    ops.append(OpDescriptor(
                        op_id=f"layer_{layer_id}.moe_dispatch",
                        layer_id=layer_id,
                        op_kind="moe_dispatch",
                        param_count=self._count_params(mlp),
                    ))
                else:
                    ops.append(OpDescriptor(
                        op_id=f"layer_{layer_id}.mlp_dense",
                        layer_id=layer_id,
                        op_kind="mlp_dense",
                        param_count=self._count_params(mlp),
                    ))
        return tuple(ops)

    @staticmethod
    def _find_submodule(layer: object, candidates: tuple[str, ...]) -> object | None:
        for name in candidates:
            sub = getattr(layer, name, None)
            if sub is not None:
                return sub
        return None

    @staticmethod
    def _is_moe(mlp: object) -> bool:
        # Heuristic: MoE blocks have an 'experts' attribute that is a list.
        experts = getattr(mlp, "experts", None)
        if experts is None:
            return False
        try:
            return len(experts) > 1
        except TypeError:
            return False

    def _count_params(self, module: object) -> int:
        """
        Sum the parameter counts of a module's leaf weights.

        Walks .parameters() if available; otherwise iterates submodules.
        """
        try:
            params = module.parameters()
        except AttributeError:
            return 0
        total = 0
        # mlx nn.Module.parameters() returns a tree; flatten it.
        flat = self._flatten_param_tree(params)
        for arr in flat:
            shape = getattr(arr, "shape", None)
            if shape is None:
                continue
            n = 1
            for dim in shape:
                n *= int(dim)
            total += n
        return total

    @classmethod
    def _flatten_param_tree(cls, tree: object) -> list[object]:
        """Recursively flatten dict/list parameter trees into a list of arrays."""
        out: list[object] = []
        if isinstance(tree, dict):
            for v in tree.values():
                out.extend(cls._flatten_param_tree(v))
        elif isinstance(tree, (list, tuple)):
            for v in tree:
                out.extend(cls._flatten_param_tree(v))
        else:
            # Leaf — assume it's an mx.array.
            out.append(tree)
        return out

    # ------------------------------------------------------------------
    # Tokenization.
    # ------------------------------------------------------------------
    def encode_corpus(
        self, text: str, max_sequences: int, sequence_length: int,
    ) -> tuple[Sequence[int], ...]:
        """
        Tokenize the entire text once, then chunk into sequences.

        We don't use the tokenizer's add_special_tokens here — calibration
        cares about typical activation behavior on raw text, not chat
        templates. The user can put whatever they want in the corpus file.
        """
        all_tokens = self._tokenizer.encode(text)
        # Some tokenizers return lists of ints; some return tensors. Normalize.
        if hasattr(all_tokens, "tolist"):
            all_tokens = all_tokens.tolist()

        sequences: list[Sequence[int]] = []
        cursor = 0
        while cursor < len(all_tokens) and len(sequences) < max_sequences:
            chunk = all_tokens[cursor:cursor + sequence_length]
            if len(chunk) < 8:
                # Skip degenerate tail chunks; activations from very short
                # sequences are noise.
                break
            sequences.append(tuple(int(t) for t in chunk))
            cursor += sequence_length
        return tuple(sequences)

    # ------------------------------------------------------------------
    # Activation capture.
    # ------------------------------------------------------------------
    def capture_reference(
        self, sequence: Sequence[int],
    ) -> dict[str, ActivationCapture]:
        """
        Run the FP forward pass, return per-op activations.

        v0.1 uses the residual stream after each transformer block as the
        activation for both attention and mlp_dense in that block. This is
        a coarse measurement — V2 should hook q/k/v/o_proj outputs and mlp
        sub-modules separately.
        """
        mx = self._mx
        tokens = mx.array([list(sequence)])
        # Run model forward to get hidden states. mlx-lm models accept tokens
        # and return logits, but we want intermediate states. We monkeypatch
        # the layers to record their outputs, then restore.
        captures: dict[str, ActivationCapture] = {}
        original_calls: list[tuple[object, object]] = []

        def make_recording_call(layer_id: int, original_call):
            def recording_call(*args, **kwargs):
                output = original_call(*args, **kwargs)
                # Layers return either a tensor or a (tensor, cache) tuple.
                hidden = output[0] if isinstance(output, tuple) else output
                # Mean-pool over sequence length to get one vector per layer.
                # We do this because activations are huge (seq_len x hidden)
                # and we just need a representative summary for divergence.
                pooled = hidden.mean(axis=1).flatten()
                shape = tuple(int(d) for d in pooled.shape)
                captures[f"layer_{layer_id}.attention"] = _MLXCapture(
                    array=pooled, _shape_tuple=shape,
                )
                # We use the same residual-stream pool for the MLP slot,
                # acknowledging the coarse-granularity tradeoff documented
                # at the top of this file.
                captures[f"layer_{layer_id}.mlp_dense"] = _MLXCapture(
                    array=pooled, _shape_tuple=shape,
                )
                captures[f"layer_{layer_id}.moe_dispatch"] = _MLXCapture(
                    array=pooled, _shape_tuple=shape,
                )
                return output
            return recording_call

        # Install hooks via __call__ replacement.
        for layer_id, layer in enumerate(self._layers):
            original = layer.__call__
            original_calls.append((layer, original))
            # Bind through a closure to capture layer_id correctly.
            layer.__call__ = make_recording_call(layer_id, original)

        try:
            _ = self._model(tokens)
        finally:
            # Always restore.
            for layer, original in original_calls:
                layer.__call__ = original

        # Filter out captures for op_ids that don't exist in this model
        # (e.g. moe_dispatch entries on a dense model).
        valid_op_ids = {op.op_id for op in self.discover_ops()}
        return {oid: cap for oid, cap in captures.items() if oid in valid_op_ids}

    def ablate_op(
        self, sequence: Sequence[int], ablation: OpAblation,
    ) -> ActivationCapture:
        """
        Run the forward pass with one op quantized, return its activation.
        """
        validate_bits(ablation.precision_bits)
        with self._temporarily_quantize(ablation.op_id, ablation.precision_bits):
            captures = self.capture_reference(sequence)
        return captures[ablation.op_id]

    def close(self) -> None:
        # mlx-lm doesn't expose a model.close(); we let GC handle it.
        self._model = None

    # ------------------------------------------------------------------
    # Quantization helper. Save weights, quantize-then-dequantize in place,
    # restore on exit. The MLX quantize round-trip is what produces the
    # "lower-precision-equivalent" weights we ablate against.
    # ------------------------------------------------------------------
    @contextmanager
    def _temporarily_quantize(self, op_id: str, bits: int) -> Iterator[None]:
        mx = self._mx
        layer_id, op_kind = self._parse_op_id(op_id)
        layer = self._layers[layer_id]
        target_module = self._target_module_for(layer, op_kind)
        if target_module is None:
            raise ValueError(
                f"Cannot find module for op {op_id} (layer {layer_id} "
                f"kind {op_kind}) in model {self._model_id_str}"
            )

        # Snapshot all weight arrays in the target module's subtree.
        # We collect (parent_module, attribute_name, original_array) tuples
        # so we can restore by attribute assignment.
        snapshots: list[tuple[object, str, object]] = []
        self._snapshot_weights(target_module, snapshots)

        try:
            # Apply quantize-then-dequantize round-trip. Group size of 64 is
            # the mlx default for affine quantization.
            for parent, attr, original in snapshots:
                if original.ndim < 2 or min(original.shape) < 64:
                    # mx.quantize requires 2D+ arrays with last dim >= group_size.
                    # Skip 1D biases and small tensors.
                    continue
                quantized, scales, biases = mx.quantize(
                    original, group_size=64, bits=bits,
                )
                dequantized = mx.dequantize(
                    quantized, scales, biases, group_size=64, bits=bits,
                )
                setattr(parent, attr, dequantized)
            yield
        finally:
            # Always restore originals.
            for parent, attr, original in snapshots:
                setattr(parent, attr, original)

    @staticmethod
    def _parse_op_id(op_id: str) -> tuple[int, str]:
        # Format: "layer_{N}.{op_kind}"
        try:
            layer_part, kind = op_id.split(".", 1)
            layer_id = int(layer_part[len("layer_"):])
            return layer_id, kind
        except (ValueError, IndexError):
            raise ValueError(f"Malformed op_id: {op_id!r}")

    def _target_module_for(self, layer: object, op_kind: str) -> object | None:
        if op_kind == "attention":
            return self._find_submodule(layer, ("self_attn", "attention", "attn"))
        if op_kind in ("mlp_dense", "moe_dispatch"):
            return self._find_submodule(layer, ("mlp", "feed_forward", "ffn"))
        return None

    @classmethod
    def _snapshot_weights(
        cls, module: object, out: list[tuple[object, str, object]],
    ) -> None:
        """
        Walk a module's children and collect (parent, attr, array) for every
        weight array. We do this manually rather than via .parameters() so we
        can restore by setattr later.
        """
        for attr in dir(module):
            if attr.startswith("_"):
                continue
            try:
                value = getattr(module, attr)
            except AttributeError:
                continue
            # Identify mx.array by checking for .shape and .dtype on the
            # value. Avoids hard-coupling to a specific MLX API surface.
            if hasattr(value, "shape") and hasattr(value, "dtype") and not callable(value):
                out.append((module, attr, value))
            elif hasattr(value, "parameters"):
                # Recurse into submodules.
                cls._snapshot_weights(value, out)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if hasattr(item, "parameters"):
                        cls._snapshot_weights(item, out)
