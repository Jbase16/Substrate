"""
substrate.runtime.verifier — Per-op disagreement signals for the TierController.

The verifier predicts how far off the live (low-precision) activations
have drifted from FP16. The controller uses this signal to decide whether
to escalate an op to a higher-precision tier.

Two implementations:
  - LinearProbeVerifier: learned probes from calibration's activation dumps.
  - ScriptedVerifier: test stub that emits pre-scripted values.

The probe protocol is simple: given a [hidden_dim] activation vector,
output a [0, 1] disagreement score. High = model is diverging; escalate.

================================================================================
NORMALIZATION CONTRACT
================================================================================

LinearProbeVerifier loads probes from a JSON file (format:
substrate_linear_probe_v1) and applies per-probe feature normalization.

The naive contract:

    Training:  X_norm = (X - feature_mean_train) / feature_std_train
               probe: linear(X_norm) -> y_norm

    Runtime:   hidden_norm = (hidden - feature_mean_train) / feature_std_train
               disagreement = sigmoid(probe(hidden_norm))

PROBLEM: feature_mean_train / feature_std_train are computed on the
calibration corpus. At runtime, on an out-of-distribution prompt
(different domain, different tokenizer patterns), the activations live
in a different region of feature space. Normalizing with stale stats
produces hidden_norm vectors that are huge in magnitude. The linear model
projects them to logits >> 5, sigmoid saturates at 0.99+, and ALL ops
look maximally diverged.

What the probe ACTUALLY learns at training time is something like
"how does precision-induced perturbation deviate from the typical (mean)
activation pattern at this op." That signal is invariant under the
training distribution but not under prompt-induced distribution shift.

FIX (this module): runtime renormalization. Accumulate per-op stats from
the first K hidden states observed at runtime, freeze, and use those
instead of the saved training stats. The probe weights are still trained
on training-time normalization, but the runtime stats put the hidden
state back into a normalized space that the probe can interpret.

ENABLED BY DEFAULT for LinearProbeVerifier. Disable via
runtime_stats=False to recover the old training-stats behavior.

LOGIT CLIPPING: even with renormalization, we clip pre-sigmoid logits
to [-5, +5] to prevent residual saturation. Sigmoid([-5, +5]) is
[0.0067, 0.9933], which preserves rank ordering near the extremes
while still allowing the controller to discriminate.

================================================================================
RUNTIME STATS DESIGN
================================================================================

Per op, we accumulate sufficient statistics for mean and variance using
Welford's online algorithm:

    n   = number of hidden states observed for this op
    mu  = running mean,    shape [hidden_dim]
    M2  = running sum of squared diffs from mean, shape [hidden_dim]

    on each new x:
        n   += 1
        delta = x - mu
        mu  += delta / n
        M2  += delta * (x - mu)   # uses NEW mu

    variance = M2 / (n - 1)
    std      = sqrt(variance)

Welford is numerically stable and incremental, so we never accumulate
in O(N) memory.

After K observations (default 8), the stats are FROZEN and used for
normalization. Before K, the verifier returns 0.0 (no signal — we don't
trust short-window stats). This means the controller sees no escalation
signal for the first K*num_ops verifier calls of a session, which is
acceptable: K=8 corresponds to ~8 forward passes through a layer.

reset() drops both EMA and runtime stats. Call between prompts.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol

log = logging.getLogger(__name__)


class VerifierProbe(Protocol):
    def disagreement(self, op_id: str, hidden: object) -> float:
        """
        Predict disagreement for one op given its current hidden state.

        Args:
            op_id: operation identifier (e.g. 'layer_0.attention')
            hidden: activation vector, shape [hidden_dim]. Exact type
                    (list, tuple, np.array, mx.array) depends on caller.

        Returns:
            float in [0, 1]. 0 = agree with FP16, 1 = maximally diverged.
        """
        ...

    def reset(self) -> None:
        """Reset any per-token state (e.g. EMA counters)."""
        ...


# ---------------------------------------------------------------------------
# Linear probe verifier — the real implementation.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _ProbeWeights:
    """One probe's learned parameters and normalization constants."""

    op_id: str
    weight: tuple[float, ...]              # shape: [hidden_dim]
    bias: float
    feature_mean_train: tuple[float, ...]  # shape: [hidden_dim] (training-time)
    feature_std_train: tuple[float, ...]   # shape: [hidden_dim] (training-time)
    label_mean: float
    label_std: float


@dataclass
class _RuntimeStats:
    """Per-op online statistics (Welford). Frozen after warmup."""

    n: int = 0
    mu: list = field(default_factory=list)        # running mean
    m2: list = field(default_factory=list)        # running sum of sq-diffs
    frozen_mean: list | None = None               # mean at freeze time
    frozen_std: list | None = None                # std  at freeze time

    def update(self, x: list[float]) -> None:
        """Welford online update."""
        if self.n == 0:
            self.mu = list(x)
            self.m2 = [0.0] * len(x)
            self.n = 1
            return
        self.n += 1
        new_mu = list(self.mu)
        for i, xi in enumerate(x):
            delta = xi - self.mu[i]
            new_mu[i] = self.mu[i] + delta / self.n
            # M2 update uses the NEW mean — that's the standard Welford form.
            self.m2[i] += delta * (xi - new_mu[i])
        self.mu = new_mu

    def freeze(self) -> None:
        """
        Compute mean and std from accumulated stats, freeze them. After
        freeze, normalization uses these frozen values instead of
        recomputing from running stats.
        """
        if self.n < 2:
            # Can't compute std with one sample; freeze identity transform.
            self.frozen_mean = list(self.mu) if self.mu else []
            self.frozen_std = [1.0] * len(self.mu) if self.mu else []
            return
        self.frozen_mean = list(self.mu)
        # Bessel-corrected std: std = sqrt(M2 / (n - 1))
        self.frozen_std = [
            math.sqrt(m2_i / (self.n - 1)) for m2_i in self.m2
        ]
        # Floor the std to avoid div-by-zero on dead channels.
        self.frozen_std = [
            s if s > 1e-8 else 1.0 for s in self.frozen_std
        ]


class LinearProbeVerifier:
    """
    Loads trained linear probes from a JSON file and applies them to
    predict per-op disagreement from hidden activations.

    Constructor flags:
        runtime_stats: bool, default True
            If True (recommended): accumulate per-op runtime stats from
            the first `warmup` observations, freeze, and use those for
            normalization instead of the training-time stats. Returns 0.0
            during warmup (no trustworthy signal yet).

            If False: use the training-time feature_mean/feature_std for
            all observations. This is the old behavior; only correct when
            runtime distribution matches training distribution.

        warmup: int, default 8
            Number of observations per op to accumulate before freezing
            stats and producing real disagreement output. Smaller =
            faster ramp-up but noisier stats; larger = more reliable
            stats but more "warm-up dead zone" where no signal is emitted.

        clip_logit: float, default 5.0
            Clamp pre-sigmoid logits to [-clip_logit, +clip_logit].
            sigmoid(±5) ≈ [0.0067, 0.9933], so this preserves rank
            ordering at the saturated extremes while leaving room for
            the controller to discriminate near the threshold.

        ema_alpha: float, default 0.3
            EMA smoothing on the final disagreement output. Reduces
            single-token noise.
    """

    def __init__(
        self,
        probes_path: Path | str,
        *,
        runtime_stats: bool = True,
        warmup: int = 8,
        clip_logit: float = 5.0,
        ema_alpha: float = 0.3,
    ):
        probes_path = Path(probes_path)
        if not probes_path.exists():
            raise FileNotFoundError(f"Probes file not found: {probes_path}")

        with open(probes_path) as f:
            data = json.load(f)

        if data.get("format") != "substrate_linear_probe_v1":
            raise ValueError(
                f"Unknown probes format: {data.get('format')}. "
                f"Expected substrate_linear_probe_v1."
            )

        self._model_id = data.get("model_id", "unknown")
        self._probes: dict[str, _ProbeWeights] = {}
        self._ema: dict[str, float] = {}
        self._ema_alpha = ema_alpha
        self._clip_logit = clip_logit
        self._use_runtime_stats = runtime_stats
        self._warmup = max(2, warmup)
        self._runtime_stats: dict[str, _RuntimeStats] = {}

        for op_id, probe_data in data.get("probes", {}).items():
            try:
                weight = tuple(probe_data["weight"])
                bias = float(probe_data["bias"])
                feature_mean_train = tuple(probe_data["feature_mean"])
                feature_std_train = tuple(probe_data["feature_std"])
                label_mean = float(probe_data["label_mean"])
                label_std = float(probe_data["label_std"])

                if len(feature_mean_train) != len(feature_std_train):
                    raise ValueError(
                        f"op {op_id}: feature_mean and feature_std length mismatch"
                    )
                if len(weight) != len(feature_mean_train):
                    raise ValueError(
                        f"op {op_id}: weight length != feature dimension"
                    )

                self._probes[op_id] = _ProbeWeights(
                    op_id=op_id,
                    weight=weight,
                    bias=bias,
                    feature_mean_train=feature_mean_train,
                    feature_std_train=feature_std_train,
                    label_mean=label_mean,
                    label_std=label_std,
                )
            except KeyError as e:
                raise ValueError(f"op {op_id}: missing required field {e}") from e

        log.info(
            "LinearProbeVerifier: %d probes for %s "
            "(runtime_stats=%s, warmup=%d, clip_logit=%.1f)",
            len(self._probes), self._model_id,
            self._use_runtime_stats, self._warmup, self._clip_logit,
        )

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------
    def disagreement(self, op_id: str, hidden: object) -> float:
        """
        Predict disagreement for one op.

        Behavior depends on runtime_stats flag and warmup state:
          - runtime_stats=False: always use training-time stats.
          - runtime_stats=True, n < warmup: accumulate, return 0.
          - runtime_stats=True, n == warmup: freeze stats on this call,
                                              then proceed to predict.
          - runtime_stats=True, n > warmup: predict using frozen stats.
        """
        probe = self._probes.get(op_id)
        if probe is None:
            return 0.0

        try:
            hidden_list = list(hidden)
        except TypeError:
            return 0.0

        if len(hidden_list) != len(probe.weight):
            log.warning(
                "op %s: hidden length %d != weight length %d",
                op_id, len(hidden_list), len(probe.weight),
            )
            return 0.0

        # ---- Path 1: runtime stats disabled. Use training-time stats. ----
        if not self._use_runtime_stats:
            return self._predict_with_stats(
                probe, hidden_list,
                list(probe.feature_mean_train),
                list(probe.feature_std_train),
            )

        # ---- Path 2: runtime stats enabled. ----
        rs = self._runtime_stats.setdefault(op_id, _RuntimeStats())

        # Warmup phase: accumulate, return 0.
        if rs.frozen_mean is None:
            rs.update(hidden_list)
            if rs.n >= self._warmup:
                rs.freeze()
                # First post-freeze call: fall through to prediction below.
            else:
                return 0.0

        # Prediction phase: use frozen stats.
        return self._predict_with_stats(
            probe, hidden_list, rs.frozen_mean, rs.frozen_std,
        )

    def reset(self) -> None:
        """
        Reset between prompts. Drops EMA and runtime stats. Next call
        will start a new warmup phase.
        """
        self._ema.clear()
        self._runtime_stats.clear()

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------
    def _predict_with_stats(
        self,
        probe: _ProbeWeights,
        hidden_list: list[float],
        feature_mean: list[float],
        feature_std: list[float],
    ) -> float:
        """
        Apply normalization with the supplied stats (training-time or
        runtime-frozen), run the linear projection, clip the logit, apply
        sigmoid, and EMA-smooth.
        """
        # Normalize: (x - mean) / std
        hidden_norm_dot_w = 0.0
        for i, x in enumerate(hidden_list):
            std_i = feature_std[i] if feature_std[i] > 1e-8 else 1.0
            n_i = (x - feature_mean[i]) / std_i
            hidden_norm_dot_w += probe.weight[i] * n_i

        # Linear: w · x_norm + b. Then clip, then sigmoid.
        logit = hidden_norm_dot_w + probe.bias
        if logit > self._clip_logit:
            logit = self._clip_logit
        elif logit < -self._clip_logit:
            logit = -self._clip_logit

        try:
            disagreement_raw = 1.0 / (1.0 + math.exp(-logit))
        except (ValueError, OverflowError):
            disagreement_raw = 1.0 if logit > 0 else 0.0

        # EMA smoothing.
        prev = self._ema.get(probe.op_id, disagreement_raw)
        smoothed = (
            self._ema_alpha * disagreement_raw
            + (1.0 - self._ema_alpha) * prev
        )
        self._ema[probe.op_id] = smoothed
        return smoothed


# ---------------------------------------------------------------------------
# Scripted verifier — test stub.
# ---------------------------------------------------------------------------
class ScriptedVerifier:
    """
    Test stub: returns disagreement values from a pre-scripted sequence.

    Used for testing the controller logic without needing real probes.
    The scripts dict maps op_id -> list of floats; each call pops one
    value. When exhausted, returns 0.0 (below threshold).
    """

    def __init__(self, scripts: Mapping[str, list[float]]):
        self._scripts = {k: list(v) for k, v in scripts.items()}

    def disagreement(self, op_id: str, hidden: object) -> float:
        script = self._scripts.get(op_id)
        if not script:
            return 0.0
        return script.pop(0)

    def reset(self) -> None:
        pass
