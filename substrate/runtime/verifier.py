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

LinearProbeVerifier loads probes from a JSON file (format:
substrate_linear_probe_v1) and applies per-probe feature normalization.
The critical contract:

    Training:  X_norm = (X - feature_mean) / feature_std
               y_norm = (y - label_mean) / label_std
               probe: linear(X_norm) -> y_norm

    Runtime:   hidden = [hidden_dim] pooled activation
               hidden_norm = (hidden - feature_mean) / feature_std
               logit = dot(hidden_norm, weight) + bias
               disagreement = sigmoid(logit)

If the runtime skips normalization, the probe hallucinates disagreement.
If feature_mean/feature_std aren't saved with the probe, the runtime
can't normalize and the probe is useless. This implementation enforces
that contract.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

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
    weight: tuple[float, ...]  # shape: [hidden_dim]
    bias: float
    feature_mean: tuple[float, ...]  # shape: [hidden_dim]
    feature_std: tuple[float, ...]  # shape: [hidden_dim]
    label_mean: float
    label_std: float


class LinearProbeVerifier:
    """
    Loads trained linear probes from a JSON file and applies them to
    predict per-op disagreement from hidden activations.

    Probes are trained offline on calibration's paired (FP16, ablated)
    activations. The file format is:

        {
          "format": "substrate_linear_probe_v1",
          "model_id": "...",
          "probes": {
            "layer_0.attention": {
              "weight": [...],
              "bias": 0.0,
              "feature_mean": [...],
              "feature_std": [...],
              "label_mean": 0.02,
              "label_std": 0.01,
              "train_samples": 32,
              "label_mean_measured": 0.02,
              "label_std_measured": 0.01
            },
            ...
          }
        }
    """

    def __init__(self, probes_path: Path | str):
        probes_path = Path(probes_path)
        if not probes_path.exists():
            raise FileNotFoundError(f"Probes file not found: {probes_path}")

        with open(probes_path) as f:
            data = json.load(f)

        # Validate format.
        if data.get("format") != "substrate_linear_probe_v1":
            raise ValueError(
                f"Unknown probes format: {data.get('format')}. "
                f"Expected substrate_linear_probe_v1."
            )

        self._model_id = data.get("model_id", "unknown")
        self._probes: dict[str, _ProbeWeights] = {}
        self._ema: dict[str, float] = {}
        self._ema_alpha = 0.3  # Exponential moving average smoothing.

        # Load each probe, validating normalization params are present.
        for op_id, probe_data in data.get("probes", {}).items():
            try:
                # These fields are REQUIRED and break the contract if missing.
                weight = tuple(probe_data["weight"])
                bias = float(probe_data["bias"])
                feature_mean = tuple(probe_data["feature_mean"])
                feature_std = tuple(probe_data["feature_std"])
                label_mean = float(probe_data["label_mean"])
                label_std = float(probe_data["label_std"])

                # Validate shapes.
                if len(feature_mean) != len(feature_std):
                    raise ValueError(
                        f"op {op_id}: feature_mean and feature_std length mismatch"
                    )
                if len(weight) != len(feature_mean):
                    raise ValueError(
                        f"op {op_id}: weight length != feature dimension"
                    )

                self._probes[op_id] = _ProbeWeights(
                    op_id=op_id,
                    weight=weight,
                    bias=bias,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    label_mean=label_mean,
                    label_std=label_std,
                )
            except KeyError as e:
                raise ValueError(
                    f"op {op_id}: missing required field {e}"
                ) from e

        log.info(
            "LinearProbeVerifier loaded %d probes for model %s",
            len(self._probes), self._model_id,
        )

    def disagreement(self, op_id: str, hidden: object) -> float:
        """
        Predict disagreement for one op.

        Applies feature normalization as saved in the probe, runs the
        linear model, applies sigmoid to get a [0, 1] score, and
        exponentially smooths across tokens.
        """
        probe = self._probes.get(op_id)
        if probe is None:
            return 0.0

        try:
            # Convert hidden to list of floats (works for lists, tuples, arrays).
            hidden_list = list(hidden)
            if len(hidden_list) != len(probe.weight):
                log.warning(
                    "op %s: hidden length %d != weight length %d",
                    op_id, len(hidden_list), len(probe.weight),
                )
                return 0.0
        except TypeError:
            return 0.0

        # Apply feature normalization: (X - mean) / std
        hidden_norm = [
            (h - m) / s
            for h, m, s in zip(hidden_list, probe.feature_mean, probe.feature_std)
        ]

        # Linear: dot(X_norm, w) + b
        logit = sum(w * h for w, h in zip(probe.weight, hidden_norm)) + probe.bias

        # Sigmoid to [0, 1]
        import math
        try:
            disagreement_raw = 1.0 / (1.0 + math.exp(-logit))
        except (ValueError, OverflowError):
            # logit out of range; clamp sigmoid.
            disagreement_raw = 1.0 if logit > 0 else 0.0

        # EMA smoothing to reduce noise from single-token observations.
        prev = self._ema.get(op_id, disagreement_raw)
        smoothed = self._ema_alpha * disagreement_raw + (1.0 - self._ema_alpha) * prev
        self._ema[op_id] = smoothed

        return smoothed

    def reset(self) -> None:
        """Reset EMA state between prompts."""
        self._ema.clear()


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
