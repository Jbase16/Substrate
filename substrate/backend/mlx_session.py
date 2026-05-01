"""
substrate.backend.mlx_session — End-to-end forward and generation paths.

MLXForwardSession brackets MLXOpKernel with the parts of forward that
aren't ops in Substrate's IR: embeddings (input), final norm + lm_head
(output). For v0.1 this is the seam between Substrate's op-by-op
execution and the parts of the model that run as monolithic blocks.

Three operating modes:

    forward_normal(tokens)       — Run mlx-lm's normal model() for comparison.
    forward_via_kernel(tokens)   — Run embeddings -> op-by-op via kernel ->
                                    norm/head. Single forward pass, returns
                                    logits. Used by tests that compare
                                    kernel-driven vs normal forward.
    generate_via_kernel(tokens, max_new_tokens, ...) — Real autoregressive
                                    generation. KV caches persist across
                                    tokens; controller and verifier hooks
                                    fire per-op and per-token.

Both forward paths return logits [batch, seq_len, vocab]. generate
returns a dict with the generated token list plus diagnostics.
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
# Stub-typed ScheduledOp for direct kernel calls outside a real plan.
# Tests use this; real runtime uses substrate.compiler.ir.ScheduledOp.
# ---------------------------------------------------------------------------
class _SyntheticOp:
    __slots__ = ("op_id", "layer_id", "op_kind", "tier_index")

    def __init__(self, op_id: str, layer_id: int, op_kind_value: str, tier_index: int = 0):
        self.op_id = op_id
        self.layer_id = layer_id
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
    Wraps an mlx-lm Qwen2 model with an MLXOpKernel.

    Single-prompt lifecycle. Caches reset at the start of each forward()
    or generate() call.
    """

    def __init__(
        self,
        model: object,
        weight_bank: object | None = None,
    ) -> None:
        self._mx = _import_mlx()
        self._model = model
        self._weight_bank = weight_bank
        self._embed = self._find_embedding()
        self._final_norm = self._find_final_norm()
        self._lm_head = self._find_lm_head()
        self._kernel: "MLXOpKernel | None" = None

    @property
    def kernel(self) -> "MLXOpKernel":
        if self._kernel is None:
            from substrate.backend.mlx_kernel import MLXOpKernel
            self._kernel = MLXOpKernel(
                self._model, weight_bank=self._weight_bank,
            )
        return self._kernel

    def attach_weight_bank(self, weight_bank: object) -> None:
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
        mx = self._mx
        logits = self._model(tokens)
        mx.eval(logits)
        return logits

    def forward_via_kernel(self, tokens: object) -> object:
        mx = self._mx
        self.kernel.reset_caches()
        hidden = self._embed(tokens)

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

        hidden = self._final_norm(hidden)
        logits = self._lm_head(hidden)
        mx.eval(logits)
        return logits

    # ------------------------------------------------------------------
    # Autoregressive generation. The runtime path.
    # ------------------------------------------------------------------
    def generate_via_kernel(
        self,
        tokens: object,
        max_new_tokens: int,
        *,
        controller: object | None = None,
        plan: object | None = None,
        on_op_complete: object | None = None,
        on_token: object | None = None,
        sample_strategy: str = "argmax",
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> dict:
        """
        Autoregressively generate up to max_new_tokens tokens via the kernel.

        Lifecycle:
          1. Reset KV caches once.
          2. Prefill: run the full prompt. Caches populate. Capture last-token logits.
          3. For each generation step:
             a. Sample next token from current logits.
             b. on_token callback fires.
             c. controller.end_token() fires (if controller present).
             d. Forward the new single token through the kernel. seq_len=1;
                the kernel's mask is None for seq=1, cache provides history.
             e. Capture new last-token logits.
          4. Return generated tokens + diagnostics.

        Hooks:
          controller: object with .active_op(op_id) -> ScheduledOp and
                      .end_token() -> None. When provided, kernel runs ops
                      at controller-chosen tiers; bank swaps as needed.
          plan:       object with .op_bundles (iterable of OpBundles having
                      .op_id). Required when controller is provided.
          on_op_complete(op, hidden): called after each op's execute().
                      Test verifier hook.
          on_token(token_id, logits, step): called after sampling each
                      token. Test diagnostic hook.

        Sampling:
          sample_strategy="argmax": deterministic top-1.
          sample_strategy="temperature": softmax sampling. Use seed for repro.

        Returns dict:
          tokens: list[int] of generated token IDs
          prefill_logits, final_logits: mx.array
          num_prefill_tokens, num_generated: int
        """
        mx = self._mx
        self.kernel.reset_caches()

        if controller is not None and plan is None:
            raise ValueError(
                "generate_via_kernel: controller requires plan to enumerate ops."
            )

        # Build the (layer_id, kind, op_id) list. Plan order is canonical.
        if plan is not None:
            op_ids: list[tuple[int, str, str]] = []
            for ob in plan.op_bundles:
                op_id = ob.op_id
                layer_part, kind = op_id.split(".", 1)
                layer_id = int(layer_part[len("layer_"):])
                op_ids.append((layer_id, kind, op_id))
        else:
            op_ids = []
            for layer_id in range(self.num_layers):
                op_ids.append((layer_id, "attention", f"layer_{layer_id}.attention"))
                op_ids.append((layer_id, "mlp_dense", f"layer_{layer_id}.mlp_dense"))

        # Sampling state.
        rng_key = None
        if sample_strategy == "temperature" and seed is not None:
            rng_key = mx.random.key(seed)

        def sample_next(logits_last: object) -> int:
            if sample_strategy == "argmax":
                return int(mx.argmax(logits_last).item())
            if sample_strategy == "temperature":
                if temperature == 0:
                    return int(mx.argmax(logits_last).item())
                scaled = logits_last / temperature
                nonlocal rng_key
                if rng_key is None:
                    sampled = mx.random.categorical(scaled)
                else:
                    rng_key, subkey = mx.random.split(rng_key)
                    sampled = mx.random.categorical(scaled, key=subkey)
                return int(sampled.item())
            raise ValueError(f"unknown sample_strategy: {sample_strategy}")

        def run_one_pass(input_tokens: object) -> object:
            """Push one tensor [1, seq] through embed -> ops -> norm -> head."""
            hidden = self._embed(input_tokens)
            for layer_id, kind, op_id in op_ids:
                if controller is not None:
                    op = controller.active_op(op_id)
                else:
                    op = _SyntheticOp(
                        op_id=op_id, layer_id=layer_id,
                        op_kind_value=kind, tier_index=0,
                    )
                hidden = self.kernel.execute(op, decision=None, hidden=hidden)
                if on_op_complete is not None:
                    try:
                        on_op_complete(op, hidden)
                    except Exception as e:
                        log.warning("on_op_complete callback failed: %s", e)
            hidden = self._final_norm(hidden)
            logits = self._lm_head(hidden)
            mx.eval(logits)
            return logits

        # 1) Prefill.
        prefill_logits = run_one_pass(tokens)
        last_logits = prefill_logits[0, -1, :]
        mx.eval(last_logits)
        num_prefill_tokens = int(tokens.shape[1])

        # 2) Generation loop.
        generated: list[int] = []
        for step in range(max_new_tokens):
            tok = sample_next(last_logits)
            generated.append(tok)
            if on_token is not None:
                try:
                    on_token(tok, last_logits, step)
                except Exception as e:
                    log.warning("on_token callback failed: %s", e)

            # Controller advances after sampling, before next forward.
            if controller is not None:
                controller.end_token()

            # Forward the new single token. seq_len=1; cache provides history.
            new_token_arr = mx.array([[tok]])
            new_logits = run_one_pass(new_token_arr)
            last_logits = new_logits[0, -1, :]
            mx.eval(last_logits)

        return {
            "tokens": generated,
            "prefill_logits": prefill_logits,
            "final_logits": last_logits,
            "num_prefill_tokens": num_prefill_tokens,
            "num_generated": len(generated),
        }

    def reset(self) -> None:
        """Reset session state. Call between prompts."""
        if self._kernel is not None:
            self._kernel.reset_caches()

    # ------------------------------------------------------------------
    # Module discovery. Qwen2-shaped.
    # ------------------------------------------------------------------
    def _find_embedding(self) -> object:
        candidates = [
            ("model", "embed_tokens"),
            ("model", "model", "embed_tokens"),
            ("transformer", "wte"),
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
        if hasattr(self._model, "lm_head") and callable(self._model.lm_head):
            return self._model.lm_head
        embed = self._embed
        if hasattr(embed, "as_linear") and callable(embed.as_linear):
            return embed.as_linear
        raise RuntimeError(
            "Could not locate lm_head. Expected model.lm_head or tied "
            "embeddings via embed_tokens.as_linear."
        )
