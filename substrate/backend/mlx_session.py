"""
substrate.backend.mlx_session — End-to-end forward pass.

MLXForwardSession brackets MLXOpKernel with the parts of forward that
aren't ops in Substrate's IR: embeddings (input), final norm + lm_head
(output). For v0.1 this is the seam between Substrate's op-by-op
execution and the parts of the model that run as monolithic blocks.

Usage:

    session = MLXForwardSession(model)
    logits = session.forward(tokens)        # full prompt, returns logits
                                             # for last token (or all, configurable)

The session creates and owns an MLXOpKernel internally. Reset between
prompts via session.reset(). The session is single-threaded and
single-prompt-at-a-time — no batching across prompts in v0.1.

Two operating modes:

    forward_via_kernel(tokens) — Run embeddings, then op-by-op via kernel,
                                  then norm/head. This is what we test
                                  against mlx-lm's normal forward.

    forward_normal(tokens) — Run mlx-lm's normal model(tokens) for
                              comparison. Also exposed so tests can call
                              both without re-doing the cache plumbing.

Both paths return the same shape: logits [batch, seq_len, vocab].
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from substrate.backend.mlx_kernel import MLXOpKernel

log = logging.getLogger(__name__)


def _import_mlx():
    try:
        import mlx.core as mx
    except ImportError as e:
        raise ImportError(
            "MLXForwardSession requires the 'calibration' optional extra. "
            "Install with: pip install 'substrate[calibration]'"
        ) from e
    return mx


# ---------------------------------------------------------------------------
# Stub-typed ScheduledOp for the kernel call. The real type lives in
# substrate.compiler.ir; for the test path we don't have a compiled plan,
# we just need something with .layer_id and .op_kind. A namedtuple-style
# helper keeps the kernel call signature unchanged.
# ---------------------------------------------------------------------------
class _SyntheticOp:
    """
    Minimal ScheduledOp surface for direct kernel calls outside a real plan.
    Test 0/1 use this; real runtime uses substrate.compiler.ir.ScheduledOp.
    """
    __slots__ = ("op_id", "layer_id", "op_kind", "tier_index")

    def __init__(self, op_id: str, layer_id: int, op_kind_value: str, tier_index: int = 0):
        self.op_id = op_id
        self.layer_id = layer_id
        # We need an object with a .value attribute matching the IR's OpKind.
        # The kernel only reads .value, so a tiny shim works.
        self.op_kind = _OpKindShim(op_kind_value)
        self.tier_index = tier_index


class _OpKindShim:
    __slots__ = ("value",)

    def __init__(self, value: str):
        self.value = value


# ---------------------------------------------------------------------------
# Session.
# ---------------------------------------------------------------------------
class MLXForwardSession:
    """
    Wraps an mlx-lm Qwen2 model with an MLXOpKernel and provides full
    forward-pass entry points for both kernel-driven and normal execution.

    Single-prompt lifecycle: each forward() call resets caches at start.
    For multi-token generation we'd want a different lifecycle (cache
    persists across tokens), but that's runtime-loop concern handled by
    the executor — not the session's job in v0.1.
    """

    def __init__(
        self,
        model: object,
        weight_bank: object | None = None,
    ) -> None:
        self._mx = _import_mlx()
        self._model = model
        self._weight_bank = weight_bank
        # Find embedding, final norm, and lm_head modules. mlx-lm Qwen2
        # layout is: model.model.embed_tokens, model.model.norm, model.lm_head
        # (lm_head may be tied to embed_tokens; mlx-lm handles that).
        self._embed = self._find_embedding()
        self._final_norm = self._find_final_norm()
        self._lm_head = self._find_lm_head()
        # Lazy kernel construction — defer until first kernel forward.
        self._kernel: MLXOpKernel | None = None

    @property
    def kernel(self) -> "MLXOpKernel":
        """Construct kernel on first access (lazy import)."""
        if self._kernel is None:
            from substrate.backend.mlx_kernel import MLXOpKernel
            self._kernel = MLXOpKernel(
                self._model, weight_bank=self._weight_bank,
            )
        return self._kernel

    def attach_weight_bank(self, weight_bank: object) -> None:
        """
        Attach (or replace) the kernel's weight bank.

        Useful for tests and the runtime where the bank requires the
        kernel\'s layer list to construct: build session first, build
        bank from session.kernel._layers, then attach.

        Replacing a bank mid-session is allowed but the caller is
        responsible for ensuring it doesn\'t mid-flight a forward pass.
        """
        self._weight_bank = weight_bank
        if self._kernel is not None:
            self._kernel._weight_bank = weight_bank

    @property
    def num_layers(self) -> int:
        return len(self.kernel._layers)

    # ------------------------------------------------------------------
    # Forward paths.
    # ------------------------------------------------------------------
    def forward_normal(self, tokens: object) -> object:
        """
        Run the model's normal forward pass — what mlx-lm does internally.
        Returns logits.

        This is the oracle path. We compare kernel-driven forward against
        this.
        """
        mx = self._mx
        # mlx-lm Qwen2Model.__call__(tokens) → logits [batch, seq, vocab]
        logits = self._model(tokens)
        # Force evaluation. mlx is lazy; without eval the comparison reads
        # uncomputed graph nodes which can hide errors.
        mx.eval(logits)
        return logits

    def forward_via_kernel(self, tokens: object) -> object:
        """
        Run forward op-by-op via MLXOpKernel.

        Path:
            1. Embed tokens → hidden [batch, seq, hidden_dim]
            2. For each layer: attention op, then mlp op, via kernel.
            3. Final norm → norm(hidden)
            4. lm_head → logits

        Returns logits with the same shape as forward_normal.
        """
        mx = self._mx

        # Reset kernel caches: this is a fresh prompt.
        self.kernel.reset_caches()

        # Step 1: embeddings.
        hidden = self._embed(tokens)

        # Step 2: per-layer op execution.
        for layer_id in range(self.num_layers):
            attn_op = _SyntheticOp(
                op_id=f"layer_{layer_id}.attention",
                layer_id=layer_id,
                op_kind_value="attention",
            )
            hidden = self.kernel.execute(attn_op, decision=None, hidden=hidden)

            mlp_op = _SyntheticOp(
                op_id=f"layer_{layer_id}.mlp_dense",
                layer_id=layer_id,
                op_kind_value="mlp_dense",
            )
            hidden = self.kernel.execute(mlp_op, decision=None, hidden=hidden)

        # Step 3: final norm.
        hidden = self._final_norm(hidden)

        # Step 4: lm_head. If the model uses tied embeddings, lm_head may
        # be the embedding's transpose; mlx-lm exposes the right callable.
        logits = self._lm_head(hidden)
        mx.eval(logits)
        return logits

    def reset(self) -> None:
        """Reset session state. Call between prompts."""
        if self._kernel is not None:
            self._kernel.reset_caches()

    # ------------------------------------------------------------------
    # Module discovery. Qwen2-shaped.
    # ------------------------------------------------------------------
    def _find_embedding(self) -> object:
        """
        Locate the input embedding module. Qwen2 mlx-lm: model.model.embed_tokens.
        """
        # Walk common paths.
        candidates = [
            ("model", "embed_tokens"),
            ("model", "model", "embed_tokens"),
            ("transformer", "wte"),  # GPT-2 style fallback
        ]
        for path in candidates:
            obj = self._model
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok and callable(obj):
                return obj
        raise RuntimeError(
            "Could not locate embedding module. Expected model.model.embed_tokens "
            "for Qwen2-family models."
        )

    def _find_final_norm(self) -> object:
        """
        Locate the final RMSNorm before lm_head. Qwen2: model.model.norm.
        """
        candidates = [
            ("model", "norm"),
            ("model", "model", "norm"),
            ("transformer", "ln_f"),
        ]
        for path in candidates:
            obj = self._model
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok and callable(obj):
                return obj
        raise RuntimeError(
            "Could not locate final norm. Expected model.model.norm "
            "for Qwen2-family models."
        )

    def _find_lm_head(self) -> object:
        """
        Locate the language modeling head.

        Qwen2 may tie embed_tokens with lm_head (no separate lm_head
        module); in that case mlx-lm exposes a callable that does the
        right thing. We probe for both arrangements.
        """
        # Direct lm_head module.
        if hasattr(self._model, "lm_head") and callable(self._model.lm_head):
            return self._model.lm_head
        # Qwen2 with tied embeddings: model.lm_head may not exist as a
        # module; instead the model's __call__ does embedding-as-output.
        # In that case we need to use the embedding's transpose.
        # mlx-lm's Qwen2 typically does:
        #   if config.tie_word_embeddings:
        #       return self.model.embed_tokens.as_linear(hidden)
        # Reproduce that:
        embed = self._embed
        if hasattr(embed, "as_linear") and callable(embed.as_linear):
            return embed.as_linear
        raise RuntimeError(
            "Could not locate lm_head. Expected model.lm_head or tied "
            "embeddings via embed_tokens.as_linear."
        )
