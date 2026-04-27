from __future__ import annotations
import logging
from collections import Counter, deque
from dataclasses import dataclass
from typing import Mapping
from substrate.compiler.ir import TensorRef

log = logging.getLogger(__name__)

class SingleSharedExpert:
    def __init__(self, shared_expert_tensor_id: str):
        self._tensor_id = shared_expert_tensor_id
    def shared_expert_for(self, missed_op_id: str, missed_tensor: str) -> TensorRef:
        return TensorRef(self._tensor_id)

class PerLayerSharedExpert:
    def __init__(self, mapping: Mapping[int, str]):
        self._mapping = dict(mapping)
    def shared_expert_for(self, missed_op_id: str, missed_tensor: str) -> TensorRef:
        try:
            layer_id = int(missed_tensor.split(".")[0][len("block_"):])
        except (ValueError, IndexError):
            raise ValueError(f"Cannot parse layer id from {missed_tensor!r}")
        if layer_id not in self._mapping:
            raise KeyError(f"No shared expert for layer {layer_id}")
        return TensorRef(self._mapping[layer_id])

@dataclass
class _LayerHistory:
    layer_id: int
    window: deque
    counts: Counter
    def observe(self, expert_indices):
        if len(self.window) == self.window.maxlen:
            old = self.window.popleft()
            self.counts.subtract(old)
            self.counts += Counter()
        self.window.append(list(expert_indices))
        self.counts.update(expert_indices)
    def top_k(self, k):
        return tuple(idx for idx, _ in self.counts.most_common(k))

class ExpertResidencyHistory:
    def __init__(self, window_tokens: int = 128):
        self._window_tokens = window_tokens
        self._layers: dict[int, _LayerHistory] = {}
    def observe(self, layer_id: int, experts: tuple):
        if layer_id not in self._layers:
            self._layers[layer_id] = _LayerHistory(layer_id, deque(maxlen=self._window_tokens), Counter())
        self._layers[layer_id].observe(experts)
    def top_k(self, layer_id: int, k: int) -> tuple:
        hist = self._layers.get(layer_id)
        return hist.top_k(k) if hist else ()
    def reset(self):
        self._layers.clear()

def expert_skeleton_id(layer_id: int, expert_index: int) -> str:
    return f"block_{layer_id}.expert_{expert_index}.skeleton"

def expert_residual_id(layer_id: int, expert_index: int, residual_tier: int) -> str:
    return f"block_{layer_id}.expert_{expert_index}.residual_{residual_tier}"

def parse_expert_id(tensor_id: str) -> tuple[int, int]:
    parts = tensor_id.split(".")
    if len(parts) < 3 or not parts[0].startswith("block_") or not parts[1].startswith("expert_"):
        raise ValueError(f"Not an expert tensor id: {tensor_id!r}")
    return int(parts[0][len("block_"):]), int(parts[1][len("expert_"):])
