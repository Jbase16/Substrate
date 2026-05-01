# Benchmark Results — 2026-04-30

## Run

- Model: `mlx-community/Qwen2.5-1.5B-Instruct-bf16`
- Probes: trained on Pride and Prejudice (32 sequences × 512 tokens)
- Prompts: 50 prompts, 5 categories (technical / reasoning / factual / instruction-following / casual)
- Tokens generated per prompt per mode: 16
- Total runtime: 7m18s (8.8s per prompt)
- Probes run with: `runtime_stats=True, warmup=4, clip_logit=5.0`
- EscalationPolicy: `consecutive_hits_for_tier_2=3, persistence_tokens=8, max_concurrent_escalations=8, enable_demotion=True`

## Results

```
mode           kl_mean  top5_overlap  argmax_match  mean_active_tier  kl_delta   argmax_delta
tier0 (4-bit)  7.1851   0.4350        0.4500        0.000             —          —
adaptive@0.50  7.3808   0.4423        0.4462        0.936             -0.1958    -0.0038
adaptive@0.55  7.3280   0.4450        0.4400        0.646             -0.1430    -0.0100
adaptive@0.60  7.5844   0.4333        0.4300        0.368             -0.3993    -0.0200
adaptive@0.65  7.4106   0.4293        0.4325        0.187             -0.2256    -0.0175
adaptive@0.70  7.3671   0.4290        0.4313        0.078             -0.1820    -0.0187
```

All metrics measured against an FP16 reference (fresh model, no bank).

`kl_delta` is `tier0_kl - adaptive_kl`. Positive = adaptive closer to FP16. Negative = adaptive farther.

`argmax_delta` is `adaptive_argmax_match - tier0_argmax_match`. Positive = adaptive matches FP16 argmax more often. Negative = adaptive matches less often.

## Verdict

**The kill condition fires.** `kl_delta < 0` and `argmax_delta < 0` for every threshold while `mean_active_tier > 0`. The system pays real precision cost for output that is farther from FP16 than the cheap uniform-4-bit baseline.

## Observations

1. **Uniform tier 0 (4-bit) beats every adaptive configuration on KL.** Selective escalation introduces layer-to-layer precision mismatch that hurts more than uniform low precision does. Mixed-precision is not automatically better than uniform low precision.

2. **`kl_delta` is non-monotonic in threshold.** If the verifier were producing useful signal, high-confidence escalations (threshold 0.70 — only the most confident probes firing) should be most beneficial. Instead the curve goes -0.196, -0.143, -0.399, -0.226, -0.182 across thresholds 0.50, 0.55, 0.60, 0.65, 0.70. The probe is not ranking ops by "would-benefit-from-precision."

3. **`argmax_match` degrades roughly monotonically as escalation increases.** More tier-2 ops = worse argmax agreement with FP16. Selective precision *introduces* errors relative to uniform 4-bit.

## What this rules out

The control loop is not the problem. All 50 prompts ran to completion. Bank swaps happened. KV caches survived. The kernel correctly executed at controller-chosen tiers. RV1 (autoregressive verifier loop test) had already shown the chain works mechanically; this benchmark confirms it at scale.

The problem is upstream: **the probe's signal does not correlate with "this op needs more precision."** It correlates with something else (likely feature drift from the calibration corpus distribution), and that something else does not predict where escalation pays off.

## Next investigative step

Not policy tuning. Not a more sophisticated state machine. The next question is empirical:

> **Does probe disagreement correlate with measured quantization error per op per prompt on this benchmark set?**

To answer: for each of the 50 prompts, run forward at FP16 and capture per-op activations. Run again at tier 0 (4-bit) and capture per-op activations. Measure cosine distance per op per prompt between the two — that is the *true* quantization error the probe is trying to predict. Then correlate against the probe's disagreement value for that op-prompt.

If the correlation is positive: probe is reading the right signal but the policy is misusing it. Fix the policy.

If the correlation is near zero or negative: probe is reading something else entirely. Probe architecture or training methodology is the problem. (Likely candidates: delta features, mixed-corpus training, whitened features.)

Anything before that diagnostic is hand-waving.
