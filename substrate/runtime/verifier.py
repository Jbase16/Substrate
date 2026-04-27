from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Protocol

class VerifierProbe(Protocol):
    def disagreement(self, op_id: str, hidden: object) -> float: ...
    def reset(self) -> None: ...

@dataclass(frozen=True)
class LinearProbeWeights:
    op_id: str
    weight: tuple[float, ...]
    bias: float
    scale: float

class LinearProbeVerifier:
    def __init__(self, probes: Mapping[str, LinearProbeWeights], backend=None):
        self._probes = dict(probes)
        self._backend = backend
        self._ema: dict[str, float] = {}
        self._ema_alpha = 0.3

    def disagreement(self, op_id: str, hidden: object) -> float:
        probe = self._probes.get(op_id)
        if probe is None:
            return 0.0
        try:
            raw = sum(w * float(h) for w, h in zip(probe.weight, hidden, strict=False)) + probe.bias
        except TypeError:
            return 0.0
        scored = max(0.0, raw) * probe.scale
        prev = self._ema.get(op_id, scored)
        smoothed = self._ema_alpha * scored + (1.0 - self._ema_alpha) * prev
        self._ema[op_id] = smoothed
        return smoothed

    def reset(self) -> None:
        self._ema.clear()

class ScriptedVerifier:
    """Test stub: returns disagreement from a per-op script list."""
    def __init__(self, scripts: Mapping[str, list[float]]):
        self._scripts = {k: list(v) for k, v in scripts.items()}

    def disagreement(self, op_id: str, hidden: object) -> float:
        script = self._scripts.get(op_id)
        if not script:
            return 0.0
        return script.pop(0)

    def reset(self) -> None:
        pass
