"""
scripts/calibrate.py — Calibration CLI for Substrate.

Usage:

    # Real calibration with mlx-lm (requires substrate[calibration] extra):
    python -m scripts.calibrate \\
        --model mlx-community/Qwen2.5-1.5B-Instruct \\
        --corpus ./calibration_corpus.txt \\
        --max-sequences 64 \\
        --sequence-length 512 \\
        --precisions 2,3,4,6 \\
        --metric cosine_distance \\
        --output-root ./calibration

    # Smoke test with synthetic backend (no MLX required):
    python -m scripts.calibrate \\
        --backend synthetic \\
        --corpus ./calibration_corpus.txt \\
        --max-sequences 8 \\
        --sequence-length 128 \\
        --output-root ./calibration

Output: calibration/{model_id}/{timestamp}/{calibration,run_config,metrics_summary}.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from substrate.calibration.metrics import available_metrics
from substrate.calibration.runner import CalibrationRunner, RunnerOptions
from substrate.calibration.schema import (
    make_run_dir,
    write_calibration_run,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="substrate-calibrate",
        description=(
            "Run a calibration sweep, producing the (layer, op_kind, precision) "
            "loss table that Substrate's HybridQualityEstimator consumes."
        ),
    )
    p.add_argument(
        "--backend", default="mlx-lm",
        choices=("mlx-lm", "synthetic"),
        help="Calibration backend. mlx-lm requires the [calibration] extra.",
    )
    p.add_argument(
        "--model",
        help="HuggingFace model id (mlx-lm backend only). "
             "E.g. mlx-community/Qwen2.5-1.5B-Instruct",
    )
    p.add_argument(
        "--corpus", required=True,
        help="Path to a plain text file containing the calibration corpus.",
    )
    p.add_argument(
        "--max-sequences", type=int, default=64,
        help="Max number of sequences to draw from the corpus.",
    )
    p.add_argument(
        "--sequence-length", type=int, default=512,
        help="Token count per sequence.",
    )
    p.add_argument(
        "--precisions", default="2,3,4,6",
        help="Comma-separated list of precision bits to ablate. "
             "Supported: 2,3,4,6,8.",
    )
    p.add_argument(
        "--op-kinds", default="attention,mlp_dense,moe_dispatch",
        help="Comma-separated op kinds to calibrate. Unknown kinds are skipped.",
    )
    p.add_argument(
        "--metric", default="cosine_distance",
        choices=available_metrics(),
        help="Divergence metric.",
    )
    p.add_argument(
        "--output-root", default="./calibration",
        help="Directory under which {model_id}/{timestamp}/ will be created.",
    )
    p.add_argument(
        "--synthetic-layers", type=int, default=4,
        help="Number of layers for synthetic backend (ignored for mlx-lm).",
    )
    p.add_argument(
        "--synthetic-sensitive", default="",
        help="Comma-separated layer ids treated as sensitive in synthetic backend.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="DEBUG logging.",
    )
    return p


def _parse_int_list(s: str) -> tuple[int, ...]:
    if not s.strip():
        return ()
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _make_backend(args: argparse.Namespace):
    if args.backend == "mlx-lm":
        if not args.model:
            print("--model is required for mlx-lm backend", file=sys.stderr)
            sys.exit(2)
        # Lazy import: only pull MLX when actually needed.
        from substrate.calibration.mlxlm_backend import MLXLMBackend
        return MLXLMBackend(args.model)
    elif args.backend == "synthetic":
        from substrate.calibration.synthetic_backend import SyntheticBackend
        sensitive = _parse_int_list(args.synthetic_sensitive)
        return SyntheticBackend(
            model_id=args.model or "synthetic-test-model",
            num_layers=args.synthetic_layers,
            sensitive_layers=sensitive,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    corpus_path = Path(args.corpus)
    if not corpus_path.is_file():
        print(f"Corpus file not found: {corpus_path}", file=sys.stderr)
        return 2
    corpus_text = corpus_path.read_text(encoding="utf-8")
    if not corpus_text.strip():
        print(f"Corpus file is empty: {corpus_path}", file=sys.stderr)
        return 2

    options = RunnerOptions(
        precisions=_parse_int_list(args.precisions),
        op_kinds=tuple(s.strip() for s in args.op_kinds.split(",") if s.strip()),
        metric_name=args.metric,
        max_sequences=args.max_sequences,
        sequence_length=args.sequence_length,
    )

    backend = _make_backend(args)
    try:
        runner = CalibrationRunner(backend=backend, options=options)
        output = runner.run(corpus_text=corpus_text, corpus_path=corpus_path.resolve())
    finally:
        backend.close()

    run_dir = make_run_dir(args.output_root, output.config.model_id)
    paths = write_calibration_run(output, run_dir)

    print(f"\nCalibration written to: {run_dir}")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    summary = output.summary()
    print(f"\nSummary:")
    print(f"  cells: {summary['num_cells']}")
    print(f"  total samples: {summary['total_samples']}")
    print(f"  global mean loss: {summary['global_mean_loss']:.4f}")
    print(f"  loss range: [{summary['global_min_loss']:.4f}, {summary['global_max_loss']:.4f}]")
    print(f"  by op_kind: {summary['mean_loss_by_op_kind']}")
    print(f"  by precision: {summary['mean_loss_by_precision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
