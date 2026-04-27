from __future__ import annotations
import json, logging, os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping
from substrate.compiler.ir import TensorMetadata

log = logging.getLogger(__name__)
FORMAT_VERSION = 1

@dataclass(frozen=True)
class StoreEntry:
    tensor_id: str
    path: Path
    shape: tuple
    dtype: str
    bytes_count: int
    layer_id: int
    tier_index: int
    is_skeleton: bool

class QuantizedStore:
    MANIFEST_NAME = "manifest.json"
    TENSORS_DIR = "tensors"

    def __init__(self, root):
        self.root = Path(root)
        self._manifest_path = self.root / self.MANIFEST_NAME
        self._tensor_dir = self.root / self.TENSORS_DIR
        self._entries: dict[str, StoreEntry] = {}
        self._model_id = None
        if self._manifest_path.exists():
            self._load_manifest()

    @property
    def model_id(self) -> str:
        if self._model_id is None:
            raise RuntimeError(f"Store at {self.root} has no manifest yet")
        return self._model_id

    def __contains__(self, tensor_id): return tensor_id in self._entries
    def __iter__(self) -> Iterator[StoreEntry]: return iter(self._entries.values())

    def get(self, tensor_id: str) -> StoreEntry:
        try:
            return self._entries[tensor_id]
        except KeyError:
            raise KeyError(f"Tensor not in store: {tensor_id}")

    def read_bytes(self, tensor_id: str) -> bytes:
        entry = self.get(tensor_id)
        with open(entry.path, "rb") as f:
            data = f.read()
        if len(data) != entry.bytes_count:
            raise IOError(f"Tensor {tensor_id}: corrupt store")
        return data

    def to_catalog(self) -> dict[str, TensorMetadata]:
        return {e.tensor_id: TensorMetadata(
            tensor_id=e.tensor_id, bytes_in_ram=e.bytes_count, bytes_on_ssd=e.bytes_count,
            is_skeleton=e.is_skeleton, layer_id=e.layer_id, tier_index=e.tier_index,
        ) for e in self._entries.values()}

    def initialize(self, model_id: str):
        self._tensor_dir.mkdir(parents=True, exist_ok=True)
        self._model_id = model_id
        self._entries = {}
        self._write_manifest()

    def write_tensor(self, tensor_id, data, shape, dtype, layer_id, tier_index, is_skeleton):
        if self._model_id is None:
            raise RuntimeError("Call initialize() before writing tensors")
        path = self._tensor_dir / f"{tensor_id}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        entry = StoreEntry(tensor_id, path, tuple(shape), dtype, len(data), layer_id, tier_index, is_skeleton)
        self._entries[tensor_id] = entry
        self._write_manifest()
        return entry

    def _load_manifest(self):
        with open(self._manifest_path) as f:
            data = json.load(f)
        if data.get("format_version") != FORMAT_VERSION:
            raise IOError(f"format_version mismatch")
        self._model_id = data["model_id"]
        self._entries = {}
        for tid, meta in data["tensors"].items():
            self._entries[tid] = StoreEntry(
                tid, self._tensor_dir / f"{tid}.bin", tuple(meta["shape"]),
                meta["dtype"], meta["bytes"], meta["layer_id"], meta["tier_index"], meta["is_skeleton"],
            )

    def _write_manifest(self):
        data = {"model_id": self._model_id, "format_version": FORMAT_VERSION, "tensors": {
            tid: {"shape": list(e.shape), "dtype": e.dtype, "bytes": e.bytes_count,
                  "layer_id": e.layer_id, "tier_index": e.tier_index, "is_skeleton": e.is_skeleton}
            for tid, e in self._entries.items()
        }}
        tmp = self._manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, self._manifest_path)
