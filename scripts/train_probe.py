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

    def build_xy(self, fp16: Any, ablated: dict[int, Any]):
        """
        Stack across precisions into one (X, y) pair for fitting.
        Each precision contributes num_samples rows.
        Returns numpy arrays so the caller can split before fitting.
        """
        np = self.np
        X_list, y_list = [], []
        for bits, ablated_arr in ablated.items():
            y = self._cosine_distance_rowwise(fp16, ablated_arr)
            X_list.append(ablated_arr)
            y_list.append(y)
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        return X, y

    def _fit_arrays(self, X, y) -> dict[str, Any]:
        """
        Fit ridge regression on raw (X, y) arrays. Computes normalization
        from the data passed in — important for honest eval, where the
        held-out test set must be normalized using TRAIN statistics, not
        its own statistics.
        """
        np = self.np
        n_features = X.shape[1]

        # Normalize features using THIS data's statistics.
        feature_mean = np.mean(X, axis=0)
        feature_std = np.std(X, axis=0)
        feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
        X_norm = (X - feature_mean) / feature_std

        # Normalize labels using THIS data's statistics.
        label_mean_measured = float(np.mean(y))
        label_std_measured = float(np.std(y))
        label_std = max(label_std_measured, 1e-8)
        y_norm = (y - label_mean_measured) / label_std

        # Ridge: (X^T X + lambda I) w = X^T y
        XtX = X_norm.T @ X_norm
        Xty = X_norm.T @ y_norm
        XtX_reg = XtX + self.lambda_reg * np.eye(n_features)
        try:
            w = np.linalg.solve(XtX_reg, Xty)
        except np.linalg.LinAlgError:
            log.warning(
                "Ridge regression singular; using pseudoinverse (may overfit)"
            )
            w = np.linalg.pinv(XtX_reg) @ Xty

        b = 0.0  # Ridge on normalized data naturally centers; b ≈ 0.

        return {
            "weight": w.tolist(),
            "bias": float(b),
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "label_mean": label_mean_measured,
            "label_std": label_std,
            "train_samples": X.shape[0],
            "label_mean_measured": label_mean_measured,
            "label_std_measured": label_std_measured,
        }

    def fit_probe(self, fp16: Any, ablated: dict[int, Any]) -> dict[str, Any]:
        """Fit a probe on all data. Production path."""
        X, y = self.build_xy(fp16, ablated)
        return self._fit_arrays(X, y)

    def evaluate_split(
        self,
        fp16: Any,
        ablated: dict[int, Any],
        eval_fraction: float,
        seed: int,
    ) -> dict[str, Any]:
        """
        Train/test split evaluation. Splits rows, fits on train,
        predicts on test, returns metrics + the test-set predictions.

        Critical: the test set is normalized using TRAIN statistics
        (not test statistics). This is the honest eval contract — at
        runtime the verifier sees vectors normalized by training-time
        statistics, so eval has to mirror that.

        Returns:
            {
              "n_train": int,
              "n_test": int,
              "mae": float,           # mean absolute error in raw label units
              "rmse": float,          # root mean squared error
              "pearson_r": float,     # correlation predicted vs actual
              "label_mean": float,    # baseline: predicting label_mean has MAE = mean(|y - label_mean|)
              "baseline_mae": float,  # MAE of always-predict-mean baseline
              "skill": float,         # (baseline_mae - mae) / baseline_mae; >0 = probe beats baseline
            }
        """
        np = self.np
        X, y = self.build_xy(fp16, ablated)
        n = X.shape[0]
        if n < 5:
            # Too few samples to split meaningfully; return marker.
            return {"n_train": n, "n_test": 0, "mae": float("nan"),
                    "rmse": float("nan"), "pearson_r": float("nan"),
                    "label_mean": float(np.mean(y)) if n > 0 else 0.0,
                    "baseline_mae": float("nan"), "skill": float("nan")}

        # Reproducible shuffle.
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(round(n * eval_fraction)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Fit on train only.
        probe = self._fit_arrays(X_train, y_train)
        weight = np.array(probe["weight"])
        bias = probe["bias"]
        f_mean = np.array(probe["feature_mean"])
        f_std = np.array(probe["feature_std"])
        l_mean = probe["label_mean"]
        l_std = probe["label_std"]

        # Predict on test, using TRAIN normalization.
        X_test_norm = (X_test - f_mean) / f_std
        # Linear predicts y_norm; un-normalize back to raw label units.
        y_pred_norm = X_test_norm @ weight + bias
        y_pred = y_pred_norm * l_std + l_mean

        # Metrics in raw label units (cosine_distance domain).
        residual = y_pred - y_test
        mae = float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(residual ** 2)))

        # Pearson correlation. Guard against zero-variance test labels.
        if np.std(y_test) < 1e-12 or np.std(y_pred) < 1e-12:
            pearson_r = float("nan")
        else:
            pearson_r = float(np.corrcoef(y_test, y_pred)[0, 1])

        # Baseline: always predict y_train mean. Skill = how much better
        # the probe is than that. Negative skill = probe is worse than
        # constant prediction (definitely garbage).
        baseline_pred = np.full_like(y_test, l_mean)
        baseline_mae = float(np.mean(np.abs(baseline_pred - y_test)))
        if baseline_mae < 1e-12:
            skill = 0.0
        else:
            skill = (baseline_mae - mae) / baseline_mae

        return {
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "mae": mae,
            "rmse": rmse,
            "pearson_r": pearson_r,
            "label_mean": float(np.mean(y_train)),
            "baseline_mae": baseline_mae,
            "skill": skill,
        }

    @staticmethod
    def _cosine_distance_rowwise(a, b):
        """
        Row-wise cosine distance between two arrays of same shape.
        Returns: 1 - cos_sim, so distance=0 means identical, distance=2 means opposite.
        """
        np = _import_numpy()
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        cos_sim = np.sum(a_norm * b_norm, axis=1)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        return 1.0 - cos_sim


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def run_training(
    activations_dir: Path,
    output_path: Path,
    eval_split: float = 0.0,
    eval_seed: int = 42,
) -> int:
    """
    Load activations from a dump, train probes for each op, save to JSON.

    When eval_split > 0, also runs a held-out evaluation per op:
        - splits each op's rows by eval_split (random, fixed seed)
        - fits a probe on the train portion only
        - predicts on the test portion using TRAIN normalization
        - records MAE/RMSE/Pearson/skill in the saved probe under "eval"

    The PRODUCTION probe (the one used at runtime) is still fit on ALL
    data — eval is a diagnostic, not a data sacrifice.
    """
    np = _import_numpy()
    loader = ActivationDumpLoader(activations_dir)
    trainer = LinearProbeTrainer(lambda_reg=1e-3)

    ops = loader.list_ops()
    log.info("Found %d ops in activation dump", len(ops))
    if eval_split > 0:
        log.info(
            "Eval split: %.2f (held-out), seed=%d. "
            "Per-op metrics will be saved alongside each probe.",
            eval_split, eval_seed,
        )

    # Load model_id from calibration.json in the parent directory.
    cal_path = activations_dir.parent / "calibration.json"
    model_id = "unknown"
    if cal_path.exists():
        with open(cal_path) as f:
            cal = json.load(f)
            model_id = cal.get("config", {}).get("model_id", "unknown")

    probes: dict[str, Any] = {}
    eval_metrics: list[dict[str, Any]] = []

    for op_id in ops:
        try:
            data = loader.load_op(op_id)
        except Exception as e:
            log.error("Failed to load op %s: %s", op_id, e)
            continue

        fp16 = data["fp16"]
        ablated = data["ablated"]

        log.info(
            "Training probe for %s: %d samples × %d features",
            op_id, fp16.shape[0], fp16.shape[1],
        )

        # Optional held-out eval first (uses its own train subset for fitting).
        eval_result = None
        if eval_split > 0:
            try:
                eval_result = trainer.evaluate_split(
                    fp16, ablated, eval_fraction=eval_split, seed=eval_seed,
                )
                eval_result["op_id"] = op_id
                eval_metrics.append(eval_result)
            except Exception as e:
                log.error("Eval split failed for %s: %s", op_id, e)

        # Production probe: fit on ALL data.
        try:
            probe = trainer.fit_probe(fp16, ablated)
            if eval_result is not None:
                # Embed the held-out metrics next to the production probe.
                probe["eval"] = {
                    k: v for k, v in eval_result.items()
                    if k != "op_id"
                }
            probes[op_id] = probe
        except Exception as e:
            log.error("Failed to fit probe for %s: %s", op_id, e)
            continue

    # Print an eval summary if requested.
    if eval_metrics:
        _print_eval_summary(eval_metrics)

    output_schema = {
        "format": "substrate_linear_probe_v1",
        "model_id": model_id,
        "probes": probes,
    }
    if eval_metrics:
        output_schema["eval_summary"] = _eval_summary_dict(eval_metrics)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(output_schema, f, indent=2, sort_keys=True)
    import os
    os.replace(tmp, output_path)

    log.info("Wrote %d probes to %s", len(probes), output_path)
    return 0


def _eval_summary_dict(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-op eval metrics into a summary suitable for inclusion
    in probes.json.
    """
    np = _import_numpy()
    maes = np.array([m["mae"] for m in metrics if not _isnan(m["mae"])])
    skills = np.array([m["skill"] for m in metrics if not _isnan(m["skill"])])
    n_op_garbage = sum(
        1 for m in metrics
        if not _isnan(m["mae"]) and m["mae"] >= m["label_mean"]
    )
    return {
        "n_ops_evaluated": int(len(metrics)),
        "median_mae": float(np.median(maes)) if len(maes) else float("nan"),
        "mean_mae": float(np.mean(maes)) if len(maes) else float("nan"),
        "median_skill": float(np.median(skills)) if len(skills) else float("nan"),
        "n_ops_worse_than_baseline": n_op_garbage,
    }


def _print_eval_summary(metrics: list[dict[str, Any]]) -> None:
    """
    Print a human-readable summary of held-out eval. Calls out the
    failure mode the kill-signal check looks for: ops where MAE >=
    label_mean (predict-zero would be roughly as good).
    """
    metrics_sorted = sorted(
        [m for m in metrics if not _isnan(m["mae"])],
        key=lambda m: m["mae"],
    )
    if not metrics_sorted:
        print("No usable eval metrics.")
        return

    print()
    print("=" * 70)
    print("HELD-OUT EVAL SUMMARY")
    print("=" * 70)

    summary = _eval_summary_dict(metrics)
    print(f"  ops evaluated:            {summary['n_ops_evaluated']}")
    print(f"  median MAE:               {summary['median_mae']:.6f}")
    print(f"  mean MAE:                 {summary['mean_mae']:.6f}")
    print(f"  median skill:             {summary['median_skill']:+.3f}  "
          "(>0 = probe beats predict-mean baseline)")
    print(f"  ops worse than baseline:  {summary['n_ops_worse_than_baseline']}/{summary['n_ops_evaluated']}")

    print()
    print("BEST 10 ops by MAE:")
    for m in metrics_sorted[:10]:
        print(f"  {m['op_id']:<28} mae={m['mae']:.6f}  "
              f"skill={m['skill']:+.3f}  r={m['pearson_r']:+.3f}  "
              f"label_mean={m['label_mean']:.6f}")
    print()
    print("WORST 10 ops by MAE:")
    for m in metrics_sorted[-10:]:
        print(f"  {m['op_id']:<28} mae={m['mae']:.6f}  "
              f"skill={m['skill']:+.3f}  r={m['pearson_r']:+.3f}  "
              f"label_mean={m['label_mean']:.6f}")
    print()


def _isnan(x) -> bool:
    """Robust NaN check that handles non-float inputs."""
    try:
        return x != x  # NaN != NaN is True
    except Exception:
        return False


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
        "--eval-split", type=float, default=0.0,
        help=(
            "If > 0, run held-out evaluation per op with this test fraction "
            "(e.g. 0.2 = 20%% test). Production probe is still fit on ALL "
            "data; eval metrics are stored alongside each probe under \"eval\". "
            "Default 0 (no eval)."
        ),
    )
    p.add_argument(
        "--eval-seed", type=int, default=42,
        help="Random seed for the train/test split. Default 42.",
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
    return run_training(
        activations, output,
        eval_split=args.eval_split,
        eval_seed=args.eval_seed,
    )


if __name__ == "__main__":
    sys.exit(main())
