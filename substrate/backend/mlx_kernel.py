"""
substrate.backend.mlx_kernel — Per-op MLX execution.

Implements the OpKernel protocol from the runtime:

    kernel.execute(op, decision, hidden) -> hidden

against a real mlx-lm-loaded Qwen2 model. The kernel decomposes Qwen2
transformer blocks into op-sized halves so the planner's two ops per
layer (attention, mlp_dense) each map to a half-block evaluation.

============================================================================
QWEN2-SHAPED CODE BELOW THIS LINE.

This kernel is Qwen2-shaped and intentionally not architecture-generic.
It directly replicates the structure of mlx-lm's Qwen2TransformerBlock to
split it into op-sized halves. If you point this at a Llama, Mistral,
Gemma, or anything else without verifying block layout, the math will be
wrong and the failure will be silent.

For v0.1 this is correct: the IR is the abstraction boundary, calibration
already targets Qwen2 specifically, and one shaped kernel beats one
fragile generic kernel. v0.2 introduces a per-architecture kernel
registry; until then, treat this file as Qwen2-only.
============================================================================

Op semantics:

    layer_N.attention   →  h' = h + self_attn(input_layernorm(h))
    layer_N.mlp_dense   →  h' = h + mlp(post_attention_layernorm(h))

The residual add is part of the op. The norm is part of the op. We never
return "the bare attention output" — we return the post-residual hidden
state, because that's what the next op consumes.

KV caches:

    The kernel owns the cache list. It is lazily initialized on the first
    attention call of a prompt. Between prompts, the executor calls
    reset_caches() to discard old state.

    For v0.1 the caches are mlx-lm's standard cache objects (KVCache or
    similar, depending on mlx-lm version). We don't reach into them; we
    just hand them to self_attn and let mlx-lm manage them.

Mask:

    Causal mask for self-attention is computed once per call from the
    sequence length. mlx-lm provides a helper (create_attention_mask)
    that we'd use if it were stable across versions; we recompute
    inline to avoid version-shape risk.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from substrate.compiler.ir import ScheduledOp
    from substrate.runtime.fallback import FallbackDecision

log = logging.getLogger(__name__)


def _import_mlx():
    """Lazy import. Don't fail at module load if MLX isn't installed."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as e:
        raise ImportError(
            "MLXOpKernel requires the 'calibration' optional extra. "
            "Install with: pip install 'substrate[calibration]'"
        ) from e
    return mx, nn


# ---------------------------------------------------------------------------
# The kernel.
# ---------------------------------------------------------------------------
class MLXOpKernel:
    """
    Executes Substrate ops against a Qwen2 mlx-lm model.

    Construction:
        kernel = MLXOpKernel(mlx_lm_model)

    Lifecycle per prompt:
        kernel.reset_caches()
        for each op in plan.op_bundles:
            hidden = kernel.execute(op, decision, hidden)
        # final norm + lm_head are session-level concerns; see MLXForwardSession.

    The kernel does NOT do embeddings or the lm_head — those bracket the
    op loop and live in MLXForwardSession. The kernel does ONLY the
    layer-half computation specified by op.op_id.
    """

    def __init__(self, model: object) -> None:
        """
        Parameters
        ----------
        model
            An mlx-lm-loaded model (e.g. from `mlx_lm.load()`). For v0.1 must
            be a Qwen2-family model. We discover the layer list during
            construction and fail loudly if the structure is unexpected.
        """
        self._mx, self._nn = _import_mlx()
        self._model = model
        self._layers = self._discover_layers()
        # KV caches are lazily initialized on the first attention op of a
        # prompt. Set to None to indicate "not yet initialized this prompt."
        self._caches: list[object] | None = None

        log.info(
            "MLXOpKernel initialized over %d Qwen2 layers", len(self._layers),
        )

    # ------------------------------------------------------------------
    # OpKernel protocol.
    # ------------------------------------------------------------------
    def execute(
        self,
        op: "ScheduledOp",
        decision: "FallbackDecision",
        hidden: object,
    ) -> object:
        """
        Run one op. `decision` is currently ignored — fallback handling is
        session-level (executor decides what to do when a tensor misses its
        deadline; the kernel always assumes its inputs are ready).

        Returns the new hidden state (post-residual).
        """
        layer_id = op.layer_id
        if not (0 <= layer_id < len(self._layers)):
            raise IndexError(
                f"op {op.op_id}: layer_id {layer_id} out of range "
                f"(have {len(self._layers)} layers)"
            )
        layer = self._layers[layer_id]

        # Coarse op kinds map to half-block computations. Other op_kinds
        # (norm, embedding, lm_head) are not the kernel's responsibility
        # — those run at session level.
        op_kind = op.op_kind.value
        if op_kind == "attention":
            return self._execute_attention_half(layer, layer_id, hidden)
        if op_kind == "mlp_dense":
            return self._execute_mlp_half(layer, hidden)
        if op_kind == "moe_dispatch":
            return self._execute_moe_half(layer, hidden)
        raise ValueError(
            f"op {op.op_id}: kernel does not handle op_kind {op_kind!r}. "
            f"Supported: attention, mlp_dense, moe_dispatch."
        )

    # ------------------------------------------------------------------
    # Cache management.
    # ------------------------------------------------------------------
    def reset_caches(self) -> None:
        """
        Discard any per-prompt KV cache state.

        Call this between prompts. For a single forward pass on a fresh
        prompt, `reset_caches()` should be called once before the first
        execute() of that prompt.
        """
        self._caches = None

    def _ensure_caches(self) -> list[object]:
        """
        Lazy-initialize per-layer KV caches on first attention op.

        mlx-lm provides cache types via `mlx_lm.models.cache.make_prompt_cache`.
        We import lazily to avoid coupling the kernel to a specific cache
        implementation at module load.
        """
        if self._caches is None:
            try:
                from mlx_lm.models.cache import make_prompt_cache
                self._caches = make_prompt_cache(self._model)
            except ImportError:
                # Older mlx-lm versions had a different cache helper.
                # Fall back to a list of None; mlx-lm self_attn handles None.
                log.warning(
                    "mlx_lm.models.cache.make_prompt_cache not available; "
                    "using None caches (compatible with older mlx-lm)."
                )
                self._caches = [None] * len(self._layers)
        return self._caches

    # ------------------------------------------------------------------
    # Layer-half computations. Qwen2-shaped.
    # ------------------------------------------------------------------
    def _execute_attention_half(
        self, layer: object, layer_id: int, hidden: object,
    ) -> object:
        """
        Attention half of a Qwen2 transformer block:

            normed = input_layernorm(hidden)
            attn_out = self_attn(normed, mask, cache)
            return hidden + attn_out

        Mask is computed inline. Cache is taken from self._caches.
        """
        mx = self._mx
        caches = self._ensure_caches()

        normed = layer.input_layernorm(hidden)

        # Causal mask: shape [seq_len, seq_len], -inf above diagonal.
        # mlx-lm's self_attn computes attention scores [..., seq, seq] and
        # adds the mask before softmax. For a single forward pass on a
        # fresh prompt, this is the standard upper-triangular mask. For
        # cached generation (single new token), mask is None.
        seq_len = hidden.shape[1]
        if seq_len > 1:
            mask = self._make_causal_mask(seq_len, hidden.dtype)
        else:
            mask = None

        attn_out = layer.self_attn(normed, mask, caches[layer_id])
        return hidden + attn_out

    def _execute_mlp_half(self, layer: object, hidden: object) -> object:
        """
        MLP half of a Qwen2 transformer block:

            normed = post_attention_layernorm(hidden)
            mlp_out = mlp(normed)
            return hidden + mlp_out
        """
        normed = layer.post_attention_layernorm(hidden)
        mlp_out = layer.mlp(normed)
        return hidden + mlp_out

    def _execute_moe_half(self, layer: object, hidden: object) -> object:
        """
        MoE variant of the MLP half. Qwen2 dense models don't have this;
        Qwen-MoE/DeepSeek-V2 etc. do. v0.1 dense kernel includes this for
        forward-compatibility but the dense test path won't exercise it.
        """
        normed = layer.post_attention_layernorm(hidden)
        moe_out = layer.mlp(normed)  # MoE blocks use the same .mlp attribute
        return hidden + moe_out

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------
    def _make_causal_mask(self, seq_len: int, dtype: object) -> object:
        """
        Standard upper-triangular causal mask: 0 on/below diagonal,
        -inf above. Shape [1, 1, seq_len, seq_len] so it broadcasts
        across batch and heads.

        We construct it on every attention call. For long sequences this
        is wasteful; v1 doesn't optimize. v1.x can cache by (seq_len, dtype).
        """
        mx = self._mx
        # arange + comparison gives the upper-triangular pattern.
        # mask[i, j] = 0 if j <= i else -inf
        idx = mx.arange(seq_len)
        # Broadcast to [seq_len, seq_len]: row index i, col index j
        rows = idx.reshape(-1, 1)
        cols = idx.reshape(1, -1)
        # True where j > i (upper triangle, exclusive of diagonal).
        upper = cols > rows
        # Convert to additive mask: 0 on/below, -inf above.
        # mlx uses mx.where: where(cond, x, y) -> cond ? x : y
        neg_inf = mx.array(float("-inf"), dtype=dtype)
        zero = mx.array(0.0, dtype=dtype)
        mask = mx.where(upper, neg_inf, zero)
        # Shape [1, 1, S, S] for broadcasting across [batch, heads, S, S].
        return mask.reshape(1, 1, seq_len, seq_len)

    def _discover_layers(self) -> list[object]:
        """
        Find the transformer block list. Same logic as the calibration
        backend; we re-implement here to keep this file independent of
        substrate.calibration imports. The redundancy is intentional —
        the kernel doesn't depend on calibration.
        """
        candidates = [
            (("model",), "layers"),
            (("model", "model"), "layers"),
            (("transformer",), "h"),
        ]
        for path, attr in candidates:
            obj = self._model
            ok = True
            for step in path:
                if not hasattr(obj, step):
                    ok = False
                    break
                obj = getattr(obj, step)
            if not ok:
                continue
            value = getattr(obj, attr, None)
            if isinstance(value, (list, tuple)) and len(value) > 0:
                return list(value)
        raise RuntimeError(
            "MLXOpKernel could not locate the transformer layer list in "
            "the provided model. Verify it's a Qwen2-family model loaded "
            "via mlx_lm.load()."
        )
