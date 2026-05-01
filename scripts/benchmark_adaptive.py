"""
scripts/benchmark_adaptive.py — The benchmark that can falsify the system.

Three modes per prompt:

    fp16     : reference. Fresh model, no bank. Pure FP16 forward.
    tier0    : fixed lowest precision. Same bank as adaptive but bank
               stays at tier 0 the entire generation; no controller, no
               verifier.
    adaptive : verifier + controller + bank. The full runtime path.

Per (mode, prompt) we capture per-token logits across `tokens` generated
tokens. Metrics are computed against the FP16 reference logits:

    KL divergence (per token, then averaged)
    top-5 overlap
    argmax agreement

For the adaptive mode we additionally capture:

    avg active tier (mean tier_index across all op-token executions)
    mean ops at tier 2 per token
    max ops at tier 2 across all tokens

Per-prompt isolation pattern:

    1. Fresh model (no bank) -> capture fp16 trace.
    2. Build store + bank ONCE for this prompt's quantized runs.
    3. Reset bank to tier 0; run tier0 trace (no controller).
    4. For each threshold in the sweep:
         a. Reset bank to tier 0.
         b. Fresh TierController.
         c. Fresh LinearProbeVerifier (clean runtime stats).
         d. Run adaptive trace.
         e. Compute metrics.

Fresh verifier per threshold prevents any state contamination across
threshold sweeps. Reset bank between modes via WeightBank.reset_to_tier_0().

Output:
    Per-threshold table of (mean across all prompts) metrics for tier0
    vs adaptive, plus deltas. Printed to stdout and saved to JSON next
    to the run.

Kill condition for the system:
    adaptive does not improve KL/top-5/argmax vs tier0 while increasing
    average active tier. That means the controller spends precision for
    no measurable quality benefit.

Cost: 50 prompts × 16 tokens × (1 fp16 + 1 tier0 + N thresholds adaptive)
      forward passes per prompt.
      With 5 thresholds: 50 prompts × 16 tokens × 7 modes ≈ 5600 forwards.
      On Qwen2.5-1.5B: ~30 minutes total.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def _import_mlx():
    import mlx.core as mx
    import mlx_lm
    return mx, mlx_lm


# ---------------------------------------------------------------------------
# Metric helpers. Operate on lists of floats (one per logit) for portability.
# ---------------------------------------------------------------------------
def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    if s == 0:
        return [0.0] * len(logits)
    return [e / s for e in exps]


def kl_divergence(p: list[float], q: list[float]) -> float:
    """KL(p || q). Both must sum to 1.0. Clipped for numerical stability."""
    eps = 1e-12
    total = 0.0
    for pi, qi in zip(p, q):
        if pi <= 0:
            continue
        total += pi * (math.log(pi + eps) - math.log(qi + eps))
    return total


def top_k_overlap(a: list[float], b: list[float], k: int = 5) -> float:
    """
    Fraction of indices that are in the top-k of both. Returns 0..1.
    """
    # Get indices of top-k by sorting (descending). For 152k vocab this
    # is the dominant per-token cost; it's still cheap relative to forward.
    top_a = sorted(range(len(a)), key=lambda i: a[i], reverse=True)[:k]
    top_b = sorted(range(len(b)), key=lambda i: b[i], reverse=True)[:k]
    return len(set(top_a) & set(top_b)) / k


def argmax_match(a: list[float], b: list[float]) -> int:
    return 1 if a.index(max(a)) == b.index(max(b)) else 0


def mlx_to_list(arr) -> list[float]:
    """Convert mx.array to flat list of floats."""
    import mlx.core as mx
    mx.eval(arr)
    return [float(v) for v in arr.tolist()]


def mean_pool_hidden(hidden) -> list[float]:
    """Mean-pool [1, seq, hidden_dim] -> list of hidden_dim floats."""
    import mlx.core as mx
    pooled = mx.mean(hidden, axis=1)
    pooled = pooled[0]
    mx.eval(pooled)
    return [float(v) for v in pooled.tolist()]


# ---------------------------------------------------------------------------
# Per-mode trace capture.
#
# A "trace" is a list of [vocab]-length logits, one per generated token.
# Modes capture identical trace shape so they're comparable.
# ---------------------------------------------------------------------------
def capture_trace(
    session, mx, prompt_tokens, num_tokens: int,
    *,
    controller=None, plan=None, on_op_complete=None,
):
    """
    Run generation, capturing per-token logits via on_token. Returns
    (token_list, logit_traces, elapsed_seconds).
    """
    traces: list[list[float]] = []

    def on_token(token_id, logits, step):
        # logits is the [vocab] mx.array passed by generate_via_kernel.
        traces.append(mlx_to_list(logits))

    t0 = time.time()
    result = session.generate_via_kernel(
        prompt_tokens,
        max_new_tokens=num_tokens,
        controller=controller,
        plan=plan,
        on_op_complete=on_op_complete,
        on_token=on_token,
        sample_strategy="argmax",
    )
    elapsed = time.time() - t0
    return result["tokens"], traces, elapsed


# ---------------------------------------------------------------------------
# Per-prompt benchmark.
# ---------------------------------------------------------------------------
def benchmark_prompt(
    *,
    mlx, mlx_lm,
    model_id: str,
    prompt_text: str,
    num_tokens: int,
    probes_path: str,
    thresholds: list[float],
    warmup: int,
    log,
) -> dict[str, Any]:
    """
    Run all three modes for one prompt across all thresholds. Returns
    a dict with metrics for fp16/tier0/adaptive[threshold].
    """
    from substrate.backend.mlx_session import MLXForwardSession
    from substrate.backend.ram_weight_store import RAMWeightStore
    from substrate.backend.weight_bank import WeightBank
    from substrate.runtime.tier_controller import TierController
    from substrate.runtime.verifier import LinearProbeVerifier
    from substrate.compiler.ir import EscalationPolicy
    from substrate.bench import build_test_plan, build_op_tier_precisions
    mx = mlx

    # ----- Tokenize once. -----
    # Load tokenizer separately; mlx_lm.load returns (model, tokenizer).
    # We use one tokenizer for the prompt across all modes (deterministic).
    log.info("loading tokenizer for prompt encoding")
    _model_for_tok, tokenizer = mlx_lm.load(model_id)
    encoded = tokenizer.encode(prompt_text)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    prompt_tokens = mx.array([encoded])
    # Free this model; we'll load fresh ones per mode.
    del _model_for_tok

    # =====================================================================
    # MODE 1: FP16 reference. Fresh model, no bank.
    # =====================================================================
    log.info("[fp16] loading fresh model")
    model_fp, _ = mlx_lm.load(model_id)
    session_fp = MLXForwardSession(model_fp)
    log.info("[fp16] generating %d tokens", num_tokens)
    tokens_fp, traces_fp, elapsed_fp = capture_trace(
        session_fp, mx, prompt_tokens, num_tokens,
    )
    # Free this model and session.
    del session_fp, model_fp

    # =====================================================================
    # MODE 2: tier0 (fixed 4-bit). Fresh model, bank pinned at tier 0.
    # =====================================================================
    log.info("[tier0] loading fresh model")
    model_q, _ = mlx_lm.load(model_id)
    session_q = MLXForwardSession(model_q)
    layers = session_q.kernel._layers

    # Build plan and bank ONCE; reuse for tier0 + all adaptive thresholds.
    log.info("[tier0/adaptive] building plan + store + bank")
    plan_base = build_test_plan(
        model_id, layers, {0: "4bit", 1: "6bit", 2: "fp16_eq"},
    )
    precisions = build_op_tier_precisions(
        plan_base, {0: "4bit", 1: "6bit", 2: "fp16_eq"},
    )
    ops_with_tiers = {ob.op_id: [0, 1, 2] for ob in plan_base.op_bundles}
    store = RAMWeightStore(model_q, layers, ops_with_tiers, precisions)
    bank = WeightBank(model_q, layers, store, plan_base)
    session_q.attach_weight_bank(bank)

    # Generate tier0 trace. No controller -> kernel uses _SyntheticOp at
    # tier 0 for every op, bank stays at tier 0 throughout.
    log.info("[tier0] generating %d tokens", num_tokens)
    bank.reset_to_tier_0()
    tokens_t0, traces_t0, elapsed_t0 = capture_trace(
        session_q, mx, prompt_tokens, num_tokens,
    )

    # =====================================================================
    # MODE 3: adaptive, per threshold. Reuses session_q + bank.
    # =====================================================================
    adaptive_results: dict[float, dict[str, Any]] = {}

    for threshold in thresholds:
        log.info("[adaptive thr=%.2f] preparing", threshold)
        # Fresh policy with this threshold.
        policy = EscalationPolicy(
            disagreement_threshold=threshold,
            consecutive_hits_for_tier_2=3,
            persistence_tokens=8,
            max_concurrent_escalations=8,
            enable_demotion=True,
        )
        plan = dataclasses.replace(plan_base, escalation_policy=policy)

        # Reset bank to tier 0 for this run.
        bank.reset_to_tier_0()

        # Fresh controller AND fresh verifier (clean runtime stats).
        controller = TierController(plan)
        verifier = LinearProbeVerifier(
            probes_path,
            runtime_stats=True,
            warmup=warmup,
        )

        # Per-op tier execution log (for active-tier metrics).
        per_op_tiers: list[int] = []  # one entry per op-execution

        def on_op_complete(op, hidden):
            pooled = mean_pool_hidden(hidden)
            d = verifier.disagreement(op.op_id, pooled)
            controller.observe(op.op_id, d)
            per_op_tiers.append(op.tier_index)

        log.info("[adaptive thr=%.2f] generating %d tokens", threshold, num_tokens)
        tokens_ad, traces_ad, elapsed_ad = capture_trace(
            session_q, mx, prompt_tokens, num_tokens,
            controller=controller, plan=plan,
            on_op_complete=on_op_complete,
        )

        # Tier-2 ops per token: count how many ops were at tier 2 in each
        # forward pass. The on_op_complete fires once per op per pass; we
        # bucket by pass. Number of passes = prefill (1) + num_tokens.
        # To split per_op_tiers into passes, we use len(plan.op_bundles)
        # as the per-pass op count.
        ops_per_pass = len(plan.op_bundles)
        passes = []
        for i in range(0, len(per_op_tiers), ops_per_pass):
            chunk = per_op_tiers[i:i + ops_per_pass]
            if len(chunk) == ops_per_pass:
                passes.append(chunk)
        # passes[0] is prefill; passes[1:] are generation tokens.
        gen_passes = passes[1:] if len(passes) > 1 else []
        tier2_per_token = [sum(1 for t in p if t == 2) for p in gen_passes]

        adaptive_results[threshold] = {
            "tokens": tokens_ad,
            "traces": traces_ad,
            "elapsed_seconds": elapsed_ad,
            "mean_active_tier": (
                statistics.mean(per_op_tiers) if per_op_tiers else 0.0
            ),
            "mean_tier2_per_token": (
                statistics.mean(tier2_per_token) if tier2_per_token else 0.0
            ),
            "max_tier2_per_token": max(tier2_per_token) if tier2_per_token else 0,
            "final_bank_state": dict(bank.active_state()),
        }

    # =====================================================================
    # Cleanup.
    # =====================================================================
    del session_q, model_q, bank, store

    return {
        "prompt": prompt_text,
        "fp16": {
            "tokens": tokens_fp, "traces": traces_fp, "elapsed_seconds": elapsed_fp,
        },
        "tier0": {
            "tokens": tokens_t0, "traces": traces_t0, "elapsed_seconds": elapsed_t0,
        },
        "adaptive": adaptive_results,
    }


# ---------------------------------------------------------------------------
# Per-prompt metric computation against fp16 reference.
# ---------------------------------------------------------------------------
def compute_mode_metrics(
    fp16_traces: list[list[float]],
    test_traces: list[list[float]],
    num_tokens: int,
    elapsed_seconds: float,
) -> dict[str, float]:
    """
    Per-token metrics aggregated across all tokens in this prompt.
    """
    # If a mode generated fewer tokens (it shouldn't with argmax), just
    # compare the overlapping prefix.
    n = min(len(fp16_traces), len(test_traces))
    kls: list[float] = []
    top5s: list[float] = []
    argmax_hits: list[int] = []
    for ref_logits, test_logits in zip(fp16_traces[:n], test_traces[:n]):
        p = softmax(ref_logits)
        q = softmax(test_logits)
        kls.append(kl_divergence(p, q))
        top5s.append(top_k_overlap(ref_logits, test_logits, k=5))
        argmax_hits.append(argmax_match(ref_logits, test_logits))

    if not kls:
        return {
            "kl_mean": float("nan"),
            "top5_overlap": float("nan"),
            "argmax_match": float("nan"),
            "tok_per_sec": 0.0,
            "tokens_compared": 0,
        }

    return {
        "kl_mean": statistics.mean(kls),
        "top5_overlap": statistics.mean(top5s),
        "argmax_match": statistics.mean(argmax_hits),
        "tok_per_sec": (num_tokens / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
        "tokens_compared": n,
    }


# ---------------------------------------------------------------------------
# Cross-prompt aggregation.
# ---------------------------------------------------------------------------
def aggregate(per_prompt: list[dict[str, float]]) -> dict[str, float]:
    if not per_prompt:
        return {}
    keys = per_prompt[0].keys()
    out: dict[str, float] = {}
    for k in keys:
        vals = [p[k] for p in per_prompt if not (isinstance(p[k], float) and math.isnan(p[k]))]
        out[k] = statistics.mean(vals) if vals else float("nan")
    return out


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--probes", required=True)
    p.add_argument("--prompts", required=True,
                   help="Path to prompts file. One prompt per non-blank, non-comment line.")
    p.add_argument("--tokens", type=int, default=16,
                   help="Tokens to generate per prompt per mode.")
    p.add_argument("--max-prompts", type=int, default=0,
                   help="Cap number of prompts. 0 = use all.")
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.50, 0.55, 0.60, 0.65, 0.70])
    p.add_argument("--warmup", type=int, default=4,
                   help="Verifier runtime-stats warmup observations per op.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for prompt order shuffle.")
    p.add_argument("--output", default=None,
                   help="Path to write detailed JSON results. "
                        "Default: bench_results_<timestamp>.json next to probes.")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("benchmark_adaptive")

    # Load prompts.
    prompt_lines: list[str] = []
    with open(args.prompts) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                prompt_lines.append(s)

    if args.max_prompts > 0:
        prompt_lines = prompt_lines[:args.max_prompts]

    # Hard-fail: if no prompts loaded, the run will silently produce empty
    # results. Surface the failure now instead.
    if not prompt_lines:
        print(
            f"FATAL: no prompts loaded from {args.prompts!r}. "
            f"Check the file exists and contains non-comment, non-blank lines.",
            file=sys.stderr,
        )
        return 2

    # Deterministic shuffle so categories don't cluster.
    rng = random.Random(args.seed)
    rng.shuffle(prompt_lines)
    log.info("Loaded %d prompts (shuffled with seed %d)", len(prompt_lines), args.seed)
    log.info("Thresholds: %s", args.thresholds)
    log.info("Tokens per generation: %d", args.tokens)

    mlx, mlx_lm = _import_mlx()

    all_results: list[dict[str, Any]] = []

    t_start = time.time()

    for i, prompt in enumerate(prompt_lines):
        prompt_log = logging.getLogger(f"prompt_{i+1}")
        log.info(
            "=== Prompt %d/%d (elapsed %.1fs): %s ===",
            i + 1, len(prompt_lines), time.time() - t_start, prompt[:60],
        )
        try:
            result = benchmark_prompt(
                mlx=mlx, mlx_lm=mlx_lm,
                model_id=args.model,
                prompt_text=prompt,
                num_tokens=args.tokens,
                probes_path=args.probes,
                thresholds=args.thresholds,
                warmup=args.warmup,
                log=prompt_log,
            )
            all_results.append(result)
        except Exception as e:
            # Always print to stderr, regardless of log level. Silent failure
            # produced an empty results file once already. Never again.
            import traceback
            print(
                f"\nPrompt {i + 1} failed: {e!r}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            continue

    t_end = time.time()
    log.info("Total elapsed: %.1fs (%.1fs per prompt)",
             t_end - t_start, (t_end - t_start) / max(1, len(all_results)))

    # If every prompt failed, the loop logged each exception but main() would
    # otherwise continue and write a results file full of zeros. Prevent that.
    if not all_results:
        print(
            f"FATAL: 0 of {len(prompt_lines)} prompts succeeded. "
            f"The benchmark produced no data. Re-run with --verbose to see "
            f"per-prompt exceptions.",
            file=sys.stderr,
        )
        return 3

    # =====================================================================
    # Compute aggregate metrics.
    # =====================================================================
    per_prompt_tier0_metrics: list[dict[str, float]] = []
    per_prompt_adaptive_metrics: dict[float, list[dict[str, float]]] = {
        thr: [] for thr in args.thresholds
    }
    per_prompt_adaptive_extra: dict[float, list[dict[str, Any]]] = {
        thr: [] for thr in args.thresholds
    }

    for result in all_results:
        fp16 = result["fp16"]
        # tier0 metrics.
        m = compute_mode_metrics(
            fp16["traces"], result["tier0"]["traces"],
            num_tokens=args.tokens,
            elapsed_seconds=result["tier0"]["elapsed_seconds"],
        )
        per_prompt_tier0_metrics.append(m)
        # adaptive metrics per threshold.
        for thr in args.thresholds:
            ad = result["adaptive"][thr]
            m = compute_mode_metrics(
                fp16["traces"], ad["traces"],
                num_tokens=args.tokens,
                elapsed_seconds=ad["elapsed_seconds"],
            )
            per_prompt_adaptive_metrics[thr].append(m)
            per_prompt_adaptive_extra[thr].append({
                "mean_active_tier": ad["mean_active_tier"],
                "mean_tier2_per_token": ad["mean_tier2_per_token"],
                "max_tier2_per_token": ad["max_tier2_per_token"],
            })

    tier0_agg = aggregate(per_prompt_tier0_metrics)

    # =====================================================================
    # Print summary.
    # =====================================================================
    print()
    print("=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    print(f"Model:    {args.model}")
    print(f"Probes:   {args.probes}")
    print(f"Prompts:  {len(all_results)} of {len(prompt_lines)} succeeded")
    print(f"Tokens:   {args.tokens} per prompt")
    print(f"Total runtime: {t_end - t_start:.1f}s")
    print()

    # Header row.
    cols = (
        "mode",
        "kl_mean",
        "top5_overlap",
        "argmax_match",
        "tok_per_sec",
        "mean_active_tier",
        "mean_tier2",
        "max_tier2",
        "kl_delta",
        "argmax_delta",
    )
    fmt_widths = [12, 10, 12, 13, 11, 16, 10, 10, 12, 14]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, fmt_widths))
    print(header)
    print("-" * len(header))

    # tier0 row.
    row = [
        "tier0",
        f"{tier0_agg.get('kl_mean', float('nan')):.4f}",
        f"{tier0_agg.get('top5_overlap', float('nan')):.4f}",
        f"{tier0_agg.get('argmax_match', float('nan')):.4f}",
        f"{tier0_agg.get('tok_per_sec', 0.0):.2f}",
        "0.000",  # tier0 always at tier 0
        "0.0",
        "0",
        "—",  # kl_delta vs itself = 0; print dash
        "—",
    ]
    print("  ".join(c.ljust(w) for c, w in zip(row, fmt_widths)))

    # adaptive rows, one per threshold.
    final_rows: list[dict[str, Any]] = []
    for thr in args.thresholds:
        agg = aggregate(per_prompt_adaptive_metrics[thr])
        extras = per_prompt_adaptive_extra[thr]
        mean_active = statistics.mean(
            x["mean_active_tier"] for x in extras
        ) if extras else 0.0
        mean_t2 = statistics.mean(
            x["mean_tier2_per_token"] for x in extras
        ) if extras else 0.0
        max_t2 = max(
            (x["max_tier2_per_token"] for x in extras), default=0
        )
        # Deltas: positive is "better than tier0".
        # KL: smaller is better; delta_kl = tier0.kl - adaptive.kl
        kl_delta = tier0_agg.get("kl_mean", 0.0) - agg.get("kl_mean", 0.0)
        # argmax: larger is better; delta = adaptive.argmax - tier0.argmax
        argmax_delta = (
            agg.get("argmax_match", 0.0) - tier0_agg.get("argmax_match", 0.0)
        )

        row = [
            f"adaptive@{thr:.2f}",
            f"{agg.get('kl_mean', float('nan')):.4f}",
            f"{agg.get('top5_overlap', float('nan')):.4f}",
            f"{agg.get('argmax_match', float('nan')):.4f}",
            f"{agg.get('tok_per_sec', 0.0):.2f}",
            f"{mean_active:.3f}",
            f"{mean_t2:.1f}",
            f"{max_t2}",
            f"{kl_delta:+.4f}",
            f"{argmax_delta:+.4f}",
        ]
        print("  ".join(c.ljust(w) for c, w in zip(row, fmt_widths)))

        final_rows.append({
            "threshold": thr,
            **agg,
            "mean_active_tier": mean_active,
            "mean_tier2_per_token": mean_t2,
            "max_tier2_per_token": max_t2,
            "kl_improvement_vs_tier0": kl_delta,
            "argmax_delta_vs_tier0": argmax_delta,
        })

    print()
    print("Interpretation:")
    print("  kl_delta > 0      = adaptive lower KL than tier0 (BETTER, closer to FP16)")
    print("  argmax_delta > 0  = adaptive matches FP16 argmax more often than tier0 (BETTER)")
    print("  mean_active_tier  = average tier across all op-executions (cost proxy)")
    print()
    print("  Kill condition for the system: kl_delta <= 0 AND argmax_delta <= 0")
    print("  while mean_active_tier > 0. That = paying for nothing.")
    print()

    # =====================================================================
    # Write detailed JSON.
    # =====================================================================
    if args.output is None:
        ts = time.strftime("%Y%m%dT%H%M%S")
        args.output = str(Path(args.probes).parent / f"bench_results_{ts}.json")
    out_path = Path(args.output)
    # Build serializable summary; skip the raw traces (too big).
    summary = {
        "model": args.model,
        "probes": args.probes,
        "tokens": args.tokens,
        "thresholds": args.thresholds,
        "warmup": args.warmup,
        "n_prompts": len(all_results),
        "elapsed_seconds": t_end - t_start,
        "tier0": tier0_agg,
        "adaptive": final_rows,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str, sort_keys=True)
    print(f"Detailed results: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
