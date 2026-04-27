"""
substrate.calibration.mlxlm_backend — Real backend using mlx-lm.

Implements CalibrationBackend against an mlx-lm model. Captures per-op
activations via layer-wrapper substitution, and ablates ops via mx.quantize
round-trip in a context manager that restores weights even on exception.

This module imports mlx and mlx_lm at module level. It is only loaded when
the user actually invokes calibration; the calibration package's __init__
does NOT import it. Substrate's planner-only path stays MLX-free.

Why we wrap layers instead of monkey-patching __call__:
    mlx.nn.Module extends dict and uses __call__ defined on the class.
    Reassigning instance.__call__ does not affect what `layer(...)` invokes,
    because Python resolves __call__ via type(instance).__call__. We must
    replace the layer in its parent's children list — which works because
    Qwen2Model (and most mlx-lm models) store layers as a plain Python list
    (`self.layers = [TransformerBlock(...) for _ in range(N)]`).

Coarse taxonomy in v0.1:
    We measure the layer's residual-stream output as the activation for
    BOTH the attention and mlp_dense slots in that layer. This is a
    deliberate granularity tradeoff — sub-module captures (q/k/v/o_proj,
    gate/up/down) require deeper instrumentation and don't pay off for
    the planner, which currently treats both attention sub-modules
    identically. v0.2 can split them.

Why we use mlx.quantize/dequantize for ablation:
    The point of calibration is to measure how much quality degrades when
    weights are stored at lower precision. mx.quantize(W, bits=N) produces
    the lossy quantized representation; mx.dequantize reconstructs the
    "lower-precision-equivalent" floating-point weights. Running the model
    with these reconstructed weights gives us the divergence signal we need.
"""

from __future__ import annotations

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
    Storing the array (not the flattened sequence) lets the runner compute
    multiple metrics over the same capture without re-converting.
    """
    array: object               # mx.array; typed as object to avoid MLX import
    _shape_tuple: tuple[int, ...]

    def flatten(self) -> Sequence[float]:
        # Force evaluation, then convert to Python floats via tolist().
        # mx arrays are lazy by default; tolist() is the canonical way to
        # materialize them as host data in mlx 0.31+.
        try:
            return tuple(float(v) for v in self.array.flatten().tolist())
        except (AttributeError, TypeError):
            # Fallback via numpy. Never expected on mlx 0.20+, but defensive
            # in case the captured object isn't an mx.array (tests).
            import numpy as np
            return tuple(float(v) for v in np.array(self.array).flatten())

    def shape(self) -> tuple[int, ...]:
        return self._shape_tuple


# ---------------------------------------------------------------------------
# Layer wrapper used to capture residual-stream activations.
#
# The wrapper is constructed once per backend and installed/removed via the
# _record_layer_outputs context manager. It captures into a shared dict
# keyed by layer_id, which the backend's capture_reference reads after the
# forward pass completes.
# ---------------------------------------------------------------------------
def _make_recording_wrapper_class(nn_module_cls):
    """
    Build a Module subclass that delegates to a wrapped layer and records
    its output. Built lazily inside _import_mlx() so we don't import mlx.nn
    at module load time.

    We can't define this as a top-level class because it must inherit from
    mlx.nn.Module, which is unavailable until the user opts into MLX.
    """

    class _RecordingLayerWrapper(nn_module_cls):
        def __init__(self, wrapped, layer_id, capture_dict):
            super().__init__()
            # Stash the wrapped layer as an attribute. mlx.nn.Module.__setattr__
            # handles both Module children and arbitrary attributes; assigning
            # a Module here makes it part of the parameter tree (which we
            # don't want to disturb but is harmless during a single forward).
            self.wrapped = wrapped
            self._layer_id = layer_id
            self._capture_dict = capture_dict

        def __call__(self, *args, **kwargs):
            output = self.wrapped(*args, **kwargs)
            # Layer outputs in mlx-lm transformer blocks are usually just
            # the hidden state tensor (shape [batch, seq, hidden]). Some
            # variants return (hidden, cache); handle both.
            hidden = output[0] if isinstance(output, tuple) else output
            # Mean-pool over sequence to get a representative summary vector.
            # Calibration cares about per-layer behavior, not per-token.
            pooled = hidden.mean(axis=1).flatten()
            shape = tuple(int(d) for d in pooled.shape)
            capture = _MLXCapture(array=pooled, _shape_tuple=shape)
            # We populate every coarse op_kind slot for this layer; the
            # backend filters by valid op_ids before returning.
            self._capture_dict[f"layer_{self._layer_id}.attention"] = capture
            self._capture_dict[f"layer_{self._layer_id}.mlp_dense"] = capture
            self._capture_dict[f"layer_{self._layer_id}.moe_dispatch"] = capture
            return output

    return _RecordingLayerWrapper


# ---------------------------------------------------------------------------
# The backend.
# ---------------------------------------------------------------------------
class MLXLMBackend:
    """
    Calibration backend backed by mlx-lm.

    Construct with a HuggingFace model id (e.g. 'mlx-community/Qwen2.5-1.5B-Instruct-bf16').
    Models that are already quantized in the snapshot (e.g. -4bit variants)
    WILL NOT be useful for calibration — we need FP16/BF16 weights to ablate
    from. Use the FP variant of any quantized model on the Hub.
    """

    def __init__(self, model_id: str) -> None:
        self._mx, self._nn, self._mlx_lm, self._load = _import_mlx()
        self._model_id_str = model_id

        log.info("Loading model %s via mlx-lm...", model_id)
        loaded = self._load(model_id)
        # mlx_lm.load returns (model, tokenizer) in current versions; older
        # versions returned (model, tokenizer, config). Handle both.
        if len(loaded) == 2:
            self._model, self._tokenizer = loaded
        else:
            self._model, self._tokenizer = loaded[0], loaded[1]

        self._layers_parent, self._layers_attr = self._discover_layer_container()
        self._layers: list = list(getattr(self._layers_parent, self._layers_attr))
        log.info("Loaded %s with %d layers", model_id, len(self._layers))

        self._wrapper_cls = _make_recording_wrapper_class(self._nn.Module)

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
    def _discover_layer_container(self) -> tuple[object, str]:
        """
        Find the parent module and attribute name that holds the transformer
        block list. We need the parent because layer swapping happens via
        `setattr(parent, attr, new_list)`, not through the layer itself.

        Returns (parent, attribute_name). E.g. for Qwen2Model, returns
        (model.model, 'layers').
        """
        candidates = [
            (("model",), "layers"),
            (("model", "model"), "layers"),
            (("transformer",), "h"),
            ((), "layers"),
        ]
        for path, attr in candidates:
            parent = self._model
            ok = True
            for step in path:
                if not hasattr(parent, step):
                    ok = False
                    break
                parent = getattr(parent, step)
            if not ok:
                continue
            value = getattr(parent, attr, None)
            if isinstance(value, (list, tuple)) and len(value) > 0:
                return parent, attr
        raise RuntimeError(
            f"Could not locate transformer layer list in model "
            f"{self._model_id_str}. Inspect model structure and update "
            f"_discover_layer_container."
        )

    def discover_ops(self) -> tuple[OpDescriptor, ...]:
        """
        For each layer, identify the attention block and the MLP block,
        then report one OpDescriptor each. Param counts are summed across
        all sub-modules of the coarse op.
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
        experts = getattr(mlp, "experts", None)
        if experts is None:
            return False
        try:
            return len(experts) > 1
        except TypeError:
            return False

    def _count_params(self, module: object) -> int:
        try:
            params = module.parameters()
        except AttributeError:
            return 0
        total = 0
        for arr in self._flatten_param_tree(params):
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
        out: list[object] = []
        if isinstance(tree, dict):
            for v in tree.values():
                out.extend(cls._flatten_param_tree(v))
        elif isinstance(tree, (list, tuple)):
            for v in tree:
                out.extend(cls._flatten_param_tree(v))
        else:
            out.append(tree)
        return out

    # ------------------------------------------------------------------
    # Tokenization.
    # ------------------------------------------------------------------
    def encode_corpus(
        self, text: str, max_sequences: int, sequence_length: int,
    ) -> tuple[Sequence[int], ...]:
        all_tokens = self._tokenizer.encode(text)
        if hasattr(all_tokens, "tolist"):
            all_tokens = all_tokens.tolist()

        sequences: list[Sequence[int]] = []
        cursor = 0
        while cursor < len(all_tokens) and len(sequences) < max_sequences:
            chunk = all_tokens[cursor:cursor + sequence_length]
            if len(chunk) < 8:
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

        Implementation: temporarily replace each layer in model.model.layers
        with a _RecordingLayerWrapper that captures the layer's output, then
        run a forward pass. Wrappers are removed on exit (always — try/finally).
        """
        captures: dict[str, ActivationCapture] = {}
        with self._record_layer_outputs(captures):
            tokens = self._mx.array([list(sequence)])
            _ = self._model(tokens)
            # Force evaluation. mx is lazy; if we don't eval the result, the
            # captured pooled tensors are unrealized and tolist() in flatten()
            # will trigger evaluation outside the timing window — fine for
            # correctness, but better to eval here for predictability.
            self._mx.eval(*[c.array for c in captures.values()])

        # Filter to only valid op_ids (e.g. drop moe_dispatch entries on a
        # dense model).
        valid_op_ids = {op.op_id for op in self.discover_ops()}
        return {oid: cap for oid, cap in captures.items() if oid in valid_op_ids}

    @contextmanager
    def _record_layer_outputs(self, capture_dict: dict) -> Iterator[None]:
        """
        Replace each layer in the parent's `layers` attribute with a wrapper
        that records its output. Restore the original list on exit.

        The wrappers populate `capture_dict` as a side effect of __call__.
        """
        original = list(getattr(self._layers_parent, self._layers_attr))
        wrapped = [
            self._wrapper_cls(layer, layer_id, capture_dict)
            for layer_id, layer in enumerate(original)
        ]
        # Update both the parent's attribute AND our own _layers cache. The
        # parent's attribute is what the model's forward pass walks; our
        # cache is what discover_ops uses (which reads underlying layers, so
        # we point it at originals during the swap to avoid confusion).
        setattr(self._layers_parent, self._layers_attr, wrapped)
        try:
            yield
        finally:
            setattr(self._layers_parent, self._layers_attr, original)

    def ablate_op(
        self, sequence: Sequence[int], ablation: OpAblation,
    ) -> ActivationCapture:
        validate_bits(ablation.precision_bits)
        with self._temporarily_quantize(ablation.op_id, ablation.precision_bits):
            captures = self.capture_reference(sequence)
        return captures[ablation.op_id]

    def close(self) -> None:
        self._model = None

    # ------------------------------------------------------------------
    # Quantization helper. Save weights, quantize-then-dequantize in place,
    # restore on exit. Uses mx.quantize/dequantize for the precision
    # round-trip.
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

        # Snapshot weight arrays in the target subtree.
        snapshots: list[tuple[object, str, object]] = []
        self._snapshot_weights(target_module, snapshots)
        if not snapshots:
            log.warning("op %s: no weight arrays found to ablate", op_id)

        try:
            for parent, attr, original in snapshots:
                if original.ndim < 2 or min(original.shape) < 64:
                    # mx.quantize requires 2D+ arrays with last dim >= group_size.
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
            for parent, attr, original in snapshots:
                setattr(parent, attr, original)

    @staticmethod
    def _parse_op_id(op_id: str) -> tuple[int, str]:
        try:
            layer_part, kind = op_id.split(".", 1)
            return int(layer_part[len("layer_"):]), kind
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
        weight array. This is more reliable than module.parameters() because
        we need stable references for setattr-based restoration.

        mlx.nn.Module is dict-based, so we iterate via its dict keys.
        """
        # mlx.nn.Module extends dict. Iterate via items() to get (name, child)
        # pairs without including private/method attributes.
        try:
            items = module.items() if hasattr(module, "items") else []
        except Exception:
            items = []

        for attr, value in items:
            # Leaf array: weight or bias.
            if hasattr(value, "shape") and hasattr(value, "dtype") and not callable(value):
                out.append((module, attr, value))
            # Submodule: recurse.
            elif hasattr(value, "items") and callable(getattr(value, "items", None)):
                cls._snapshot_weights(value, out)
            # List of submodules (e.g. MoE experts).
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if hasattr(item, "items") and callable(getattr(item, "items", None)):
                        cls._snapshot_weights(item, out)
