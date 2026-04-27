from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Mapping
from substrate.compiler.ir import OpKind
from substrate.compiler.planner import ModelProfile, OpProfile
from substrate.models.quantized_store import QuantizedStore

log = logging.getLogger(__name__)

def load_profile_from_calibration(calibration_path) -> ModelProfile:
    path = Path(calibration_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return _profile_from_dict(data)

def load_profile_from_store_and_calibration(store: QuantizedStore, calibration_path) -> ModelProfile:
    profile = load_profile_from_calibration(calibration_path)
    missing = [f"{op.op_id}.skeleton" for op in profile.ops if f"{op.op_id}.skeleton" not in store]
    if missing:
        raise ValueError(f"QuantizedStore missing {len(missing)} skeleton tensors: {missing[:5]}")
    if profile.model_id != store.model_id:
        log.warning("Model ID mismatch: calibration=%r, store=%r", profile.model_id, store.model_id)
    return profile

def _profile_from_dict(data: Mapping) -> ModelProfile:
    missing = {"model_id", "embedding_bytes", "lm_head_bytes", "runtime_overhead_bytes", "ops"} - data.keys()
    if missing:
        raise ValueError(f"Calibration JSON missing: {missing}")
    op_kind_map = {k.value: k for k in OpKind}
    ops = []
    for raw in data["ops"]:
        kind = op_kind_map.get(raw["op_kind"])
        if kind is None:
            raise ValueError(f"Op {raw.get('op_id')}: unknown op_kind {raw.get('op_kind')!r}")
        ops.append(OpProfile(
            op_id=raw["op_id"], op_kind=kind, layer_id=int(raw["layer_id"]),
            param_count=int(raw["param_count"]),
            skeleton_compute_us=int(raw["skeleton_compute_us"]),
            full_precision_compute_us=int(raw["full_precision_compute_us"]),
            sensitivity=float(raw["sensitivity"]),
            minimum_residual_bytes_per_token=int(raw.get("minimum_residual_bytes_per_token", 0)),
            moe_top_k=int(raw.get("moe_top_k", 0)),
            moe_num_experts=int(raw.get("moe_num_experts", 0)),
        ))
    return ModelProfile(
        model_id=data["model_id"], ops=tuple(ops),
        embedding_bytes=int(data["embedding_bytes"]),
        lm_head_bytes=int(data["lm_head_bytes"]),
        runtime_overhead_bytes=int(data["runtime_overhead_bytes"]),
    )

def synthesize_dense_profile(model_id, num_layers, hidden_size=4096, sensitive_layer_ids=()):
    sens_set = set(sensitive_layer_ids)
    ops = []
    qkvo = 4 * hidden_size * hidden_size
    mlp = 3 * hidden_size * (4 * hidden_size)
    for layer in range(num_layers):
        sens = 0.7 if layer in sens_set else 0.3
        ops.append(OpProfile(op_id=f"block_{layer}.attn", op_kind=OpKind.ATTENTION, layer_id=layer,
            param_count=qkvo, skeleton_compute_us=80, full_precision_compute_us=240, sensitivity=sens))
        ops.append(OpProfile(op_id=f"block_{layer}.mlp", op_kind=OpKind.MLP_DENSE, layer_id=layer,
            param_count=mlp, skeleton_compute_us=200, full_precision_compute_us=600, sensitivity=sens))
    return ModelProfile(model_id=model_id, ops=tuple(ops),
        embedding_bytes=hidden_size * 152_000 * 2, lm_head_bytes=hidden_size * 152_000 * 2,
        runtime_overhead_bytes=1_500_000_000)

def synthesize_moe_profile(model_id, num_layers, num_experts=64, top_k=8, hidden_size=4096,
                            moe_intermediate=1408, sensitive_layer_ids=()):
    sens_set = set(sensitive_layer_ids)
    ops = []
    qkvo = 4 * hidden_size * hidden_size
    expert_params = 3 * hidden_size * moe_intermediate
    for layer in range(num_layers):
        sens = 0.7 if layer in sens_set else 0.3
        ops.append(OpProfile(op_id=f"block_{layer}.attn", op_kind=OpKind.ATTENTION, layer_id=layer,
            param_count=qkvo, skeleton_compute_us=80, full_precision_compute_us=240, sensitivity=sens))
        ops.append(OpProfile(op_id=f"block_{layer}.router", op_kind=OpKind.MOE_ROUTER, layer_id=layer,
            param_count=hidden_size * num_experts, skeleton_compute_us=10, full_precision_compute_us=15,
            sensitivity=0.9))
        ops.append(OpProfile(op_id=f"block_{layer}.moe", op_kind=OpKind.MOE_DISPATCH, layer_id=layer,
            param_count=expert_params * top_k, skeleton_compute_us=120, full_precision_compute_us=400,
            sensitivity=sens, moe_top_k=top_k, moe_num_experts=num_experts))
    return ModelProfile(model_id=model_id, ops=tuple(ops),
        embedding_bytes=hidden_size * 152_000 * 2, lm_head_bytes=hidden_size * 152_000 * 2,
        runtime_overhead_bytes=1_500_000_000)
