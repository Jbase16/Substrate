"""
scripts/train_probe.py — Train linear probes from calibration activation dumps.

A linear probe learns to predict per-op divergence (cosine distance to FP16)
from a single hidden-state vector. The probe is cheap to evaluate at runtime
and provides a real signal for the TierController to escalate on.

Input: calibration run with --save-activations (activations/ dir).
Output: probes.json next to the run.

Per-op probe:
    Input: [hidden_dim] feature vector (pooled activation)
    Output: [0, 1] disagreement score (sigmoid of linear pred)
    Training: minimize MSE vs measured cosine_distance

Normalization is critical:
    Features: (X - mean) / std
    Labels:   (y - label_mean) / label_std
    Save: mean, std, label_mean, label_std with the probe.
    Runtime: the verifier MUST apply the same transformation.

Ridge regression with L2 regularization (lambda=1e-3) handles small
sample counts without overfitting. For the smoke dump (2-4 samples per
op), regularization is the only thing stopping the fit from being pure
noise.

Schema saved to probes.json:

    {
      "format": "substrate_linear_probe_v1",
      "model_id": "...",
      "probes": {
        "layer_0.attention": {
          "weight": [float, ...],       # shape: [hidden_dim]
          "bias": float,
          "feature_mean": [float, ...], # shape: [hidden_dim]
          "feature_std": [float, ...],  # shape: [hidden_dim]
          "label_mean": float,
          "label_std": float,
          "train_samples": int,
          "label_mean_measured": float,
          "label_std_measured": float
        }
      }
    }

The "measured" fields are diagnostics: the actual mean/std of the training
labels before transformation. Useful for understanding whether the probe
saw real signal or just noise.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _import_numpy():
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "train_probe.py requires numpy. "
            "Install with: pip install numpy"
        ) from e
    return np


# ---------------------------------------------------------------------------
# Loader for activation dumps.
# ---------------------------------------------------------------------------
class ActivationDumpLoader:
    """
    Loads paired (FP16, ablated) activations from a calibration run's
    activations/ directory. Strict validation ensures data integrity.
    """

    def __init__(self, activations_dir: Path) -> None:
        self.root = Path(activations_dir)
        if not self.root.is_dir():
            raise FileNotFoundError(f"activations dir not found: {self.root}")

    def load_op(self, op_id: str) -> dict[str, Any]:
        """
        Load all activations for one op. Returns:

            {
              "fp16": np.ndarray [num_samples, hidden_dim],
              "ablated": {
                bits: np.ndarray [num_samples, hidden_dim],
                ...
              },
              "meta": {...}
            }
        """
        np = _import_numpy()
        op_dir = self.root / op_id
        if not op_dir.is_dir():
            raise FileNotFoundError(f"op dir not found: {op_dir}")

        # Load metadata.
        meta_path = op_dir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        # Validate metadata.
        if not isinstance(meta.get("hidden_dim"), int):
            raise ValueError(f"meta missing or invalid hidden_dim: {op_id}")
        if not isinstance(meta.get("num_samples"), int):
            raise ValueError(f"meta missing or invalid num_samples: {op_id}")
        if not isinstance(meta.get("precisions_present"), list):
            raise ValueError(
                f"meta missing or invalid precisions_present: {op_id}"
            )

        hidden_dim = meta["hidden_dim"]
        num_samples = meta["num_samples"]
        precisions = meta["precisions_present"]

        # Load FP16 reference.
        fp16_path = op_dir / "fp16.npy"
        if not fp16_path.exists():
            raise FileNotFoundError(f"fp16.npy not found: {op_id}")
        fp16 = np.load(fp16_path)
        if fp16.shape != (num_samples, hidden_dim):
            raise ValueError(
                f"fp16.npy shape mismatch: got {fp16.shape}, "
                f"expected ({num_samples}, {hidden_dim}) for {op_id}"
            )

        # Load ablated precisions.
        ablated: dict[int, Any] = {}
        for bits in precisions:
            bits_path = op_dir / f"bits_{bits}.npy"
            if not bits_path.exists():
                raise FileNotFoundError(f"bits_{bits}.npy not found: {op_id}")
            arr = np.load(bits_path)
            if arr.shape != (num_samples, hidden_dim):
                raise ValueError(
                    f"bits_{bits}.npy shape mismatch: got {arr.shape}, "
                    f"expected ({num_samples}, {hidden_dim}) for {op_id}"
                )
            ablated[bits] = arr

        return {
            "fp16": fp16,
            "ablated": ablated,
            "meta": meta,
        }

    def list_ops(self) -> list[str]:
        """Return all op_ids in the dump, sorted."""
        ops = [
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        return sorted(ops)


# ---------------------------------------------------------------------------
# Linear probe trainer.
# ---------------------------------------------------------------------------
class LinearProbeTrainer:
    """
    Trains one linear probe per op using ridge regression.

    For each (op, precision) pair:
        input: quantized activation (pooled, shape [hidden_dim])
        label: cosine_distance(quantized, fp16)
        train: w * (X - mean) / std + b ≈ y
        eval:  sigmoid(w * (X - mean) / std + b) ∈ [0, 1]
    """

    def __init__(self, lambda_reg: float = 1e-3) -> None:
        self.np = _import_numpy()
        self.lambda_reg = lambda_reg

    def fit_probe(self, fp16: Any, ablated: dict[int, Any]) -> dict[str, Any]:
        """
        Fit a probe for one op given FP16 reference and ablated activations.

        Returns probe weights + normalization params:
            {
              "weight": [w0, w1, ...],
              "bias": b,
              "feature_mean": [m0, m1, ...],
              "feature_std": [s0, s1, ...],
              "label_mean": float,
              "label_std": float,
              "train_samples": int,
              "label_mean_measured": float,
              "label_std_measured": float,
            }
        """
        np = self.np

        # Accumulate training data across all precisions.
        X_list = []  # [num_features]
        y_list = []  # [num_labels]

        for bits, ablated_arr in ablated.items():
            # Compute labels: cosine distance between quantized and fp16.
            # Shape: [num_samples]
            y = self._cosine_distance_rowwise(fp16, ablated_arr)
            X_list.append(ablated_arr)
            y_list.append(y)

        # Stack across precisions.
        X = np.vstack(X_list)  # [num_samples * num_precisions, hidden_dim]
        y = np.hstack(y_list)  # [num_samples * num_precisions]

        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Normalize features.
        feature_mean = np.mean(X, axis=0)
        feature_std = np.std(X, axis=0)
        # Avoid division by zero on constant features.
        feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
        X_norm = (X - feature_mean) / feature_std

        # Normalize labels.
        label_mean_measured = float(np.mean(y))
        label_std_measured = float(np.std(y))
        label_std = max(label_std_measured, 1e-8)  # Avoid zero division.
        y_norm = (y - label_mean_measured) / label_std

        # Ridge regression: (X^T X + lambda I) w = X^T y
        # We solve for w by adding lambda to the diagonal of the Gram matrix.
        XtX = X_norm.T @ X_norm
        Xty = X_norm.T @ y_norm
        XtX_reg = XtX + self.lambda_reg * np.eye(n_features)
        try:
            w = np.linalg.solve(XtX_reg, Xty)
        except np.linalg.LinAlgError:
            # Singular matrix; fall back to pseudoinverse.
            log.warning(
                "Ridge regression singular; using pseudoinverse (may overfit)"
            )
            w = np.linalg.pinv(XtX_reg) @ Xty

        # Bias: mean residual after subtracting the linear term.
        b = 0.0  # Ridge on normalized data naturally centers; b ≈ 0.

        return {
            "weight": w.tolist(),
            "bias": float(b),
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "label_mean": label_mean_measured,
            "label_std": label_std,
            "train_samples": n_samples,
            "label_mean_measured": label_mean_measured,
            "label_std_measured": label_std_measured,
        }

    @staticmethod
    def _cosine_distance_rowwise(a, b):
        """
        Row-wise cosine distance between two arrays of same shape.
        Returns: 1 - cos_sim, so distance=0 means identical, distance=2 means opposite.
        """
        np = _import_numpy()
        # Normalize each row.
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        # Row-wise dot product.
        cos_sim = np.sum(a_norm * b_norm, axis=1)
        # Clamp to [-1, 1] to avoid numerical issues.
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        # Distance = 1 - cos_sim (so 0 when identical, 2 when opposite).
        return 1.0 - cos_sim


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def run_training(activations_dir: Path, output_path: Path) -> int:
    """
    Load activations from a dump, train probes for each op, save to JSON.
    """
    np = _import_numpy()
    loader = ActivationDumpLoader(activations_dir)
    trainer = LinearProbeTrainer(lambda_reg=1e-3)

    ops = loader.list_ops()
    log.info("Found %d ops in activation dump", len(ops))

    # Load model_id from calibration.json in the parent directory.
    cal_path = activations_dir.parent / "calibration.json"
    model_id = "unknown"
    if cal_path.exists():
        with open(cal_path) as f:
            cal = json.load(f)
            model_id = cal.get("config", {}).get("model_id", "unknown")

    probes: dict[str, Any] = {}

    for op_id in ops:
        try:
            data = loader.load_op(op_id)
        except Exception as e:
            log.error("Failed to load op %s: %s", op_id, e)
            continue

        fp16 = data["fp16"]
        ablated = data["ablated"]
        meta = data["meta"]

        log.info(
            "Training probe for %s: %d samples × %d features",
            op_id, fp16.shape[0], fp16.shape[1],
        )

        try:
            probe = trainer.fit_probe(fp16, ablated)
            probes[op_id] = probe
        except Exception as e:
            log.error("Failed to fit probe for %s: %s", op_id, e)
            continue

    # Build output schema.
    output_schema = {
        "format": "substrate_linear_probe_v1",
        "model_id": model_id,
        "probes": probes,
    }

    # Write to disk.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(output_schema, f, indent=2, sort_keys=True)
    import os
    os.replace(tmp, output_path)

    log.info("Wrote %d probes to %s", len(probes), output_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Train linear probes from calibration activation dumps."
    )
    p.add_argument(
        "--activations-dir", required=True,
        help="Path to activations/ directory from a calibration run.",
    )
    p.add_argument(
        "--output", required=True,
        help="Output path for probes.json.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="DEBUG logging.",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    activations = Path(args.activations_dir)
    if not activations.is_dir():
        print(f"activations-dir not found: {activations}", file=sys.stderr)
        return 2

    output = Path(args.output)
    return run_training(activations, output)


if __name__ == "__main__":
    sys.exit(main())
