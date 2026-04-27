# Substrate

**MLX-native compiler/runtime for running oversized local models under hard memory constraints.**

Substrate treats inference memory management as a rate-distortion problem — minimizing quality loss subject to a fixed RAM budget — and solves it at compile time instead of runtime. The result is a pre-validated *maneuver envelope*: a bundle of execution tiers per operation that the runtime switches between in microseconds, with no optimization required while the user is waiting.

---

## Why Substrate

Every serious local model deployment eventually hits the same wall: the model is too large to fit in unified memory at usable precision, and the standard options are all unsatisfying.

**Static quantization** reduces quality globally and permanently. You pick a precision at load time and live with it.

**Streaming/paging approaches** (AirLLM, ZeRO-Infinity variants) move weights in and out of RAM on demand, but they plan nothing ahead — every token is a reactive scramble.

**Adaptive runtime systems** try to replan on the fly when they detect quality degradation. The problem is that replanning takes time, and the user is waiting. A system that pauses to solve an optimization problem mid-generation produces visible latency spikes.

Substrate inverts the constraint:

```
Normal approach:
  problem occurs → replan → execute

Substrate:
  compile all legal responses first → problem occurs → switch path
```

The compiler does the hard work once, offline. The runtime does only state transitions — O(1) tier switches — with no solving, no planning, and no surprises.

---

## What Substrate Does

Given a model, a RAM budget, and a quality cap, Substrate:

1. **Checks feasibility** across four axes — memory, bandwidth, latency, quality — and returns a structured report with concrete relaxation options if any axis fails.

2. **Assigns precision** per operation via greedy rate-distortion optimization: minimize quality loss subject to the RAM budget.

3. **Builds a tier ladder** for each operation: a default (cheapest valid) tier plus one or two escalation tiers at higher precision. Every tier is pre-validated against the global RAM budget.

4. **Computes an escalation pool** — the RAM headroom above the default path's peak resident set — and admits escalation tiers only if the K largest simultaneous escalations fit within the pool. The pool is a hard contract; the runtime can never exceed it.

5. **Emits a `PlanBundle`** — a fully validated, serializable artifact that the runtime executes directly.

At inference time, a `TierController` watches disagreement signals from a verifier probe. When a layer drifts outside predicted bounds, the controller escalates that operation's tier, drawing from the pool. When the signal stabilizes, the tier decays back to default. No solving. No vibes.

---

## The Architecture

### Compiler

```
ModelProfile + Budget
       │
       ▼
 Feasibility Check ──── INFEASIBLE ──→ FeasibilityReport (binding_axis + relax_options)
       │
       ▼
 Greedy RDO Fill (tier-0 precision per op)
       │
       ▼
 Build Escalation Precisions (tier-1, tier-2)
       │
       ▼
 Build Tensor Catalog (all tiers, all tensors, RAM/SSD sizes)
       │
       ▼
 Residency Assignment (skeleton always resident; residuals fit in sensitivity order)
       │
       ▼
 Pool Admission (K-largest-delta constraint enforced here)
       │
       ▼
 Build OpBundles (per-op tier ladders with prefetch/evict/fallback annotations)
       │
       ▼
    PlanBundle
```

### PlanBundle Structure

```
PlanBundle
  ├─ budget                        # RAM, SSD bandwidth, quality cap, target TPS
  ├─ tensor_catalog                # Every tensor at every tier: size in RAM, size on SSD
  ├─ op_bundles[]                  # One per operation, in topological order
  │    ├─ tiers[0]  (default)      # Cheapest valid; runs by default
  │    ├─ tiers[1]  (escalation)   # +1 precision step; costs Δ RAM from pool
  │    └─ tiers[2]  (ceiling)      # +2 steps; highest quality within pool budget
  ├─ escalation_policy             # Thresholds, persistence, max concurrent
  ├─ fallback_policy               # Deadline-miss strategy, critical drift factors
  └─ escalation_ram_pool_bytes     # Hard pool: sum of any K active escalation deltas ≤ this
```

Each `ScheduledOp` carries:
- `requires` — tensors that must be resident at op entry
- `prefetch` — SSD reads to issue during the prior op's compute window, with deadlines
- `evict_after` — tensors to free after this op completes (explicit, not implicit)
- `fallback` — what to do if a streamed tensor misses its deadline
- `peak_ram_delta_bytes` — cost above tier-0 for escalation pool accounting

### IR Invariants (validated at construction)

Every `PlanBundle` enforces at construction time:
- All tensor references resolve in the catalog
- Prefetch `start_during` precedes `deadline_before` in topological order
- Default path (all tier-0) fits within `budget.max_ram_bytes`
- Per `OpBundle`: `quality_risk` monotonically non-increasing in tier index
- Per `OpBundle`: `peak_ram_delta_bytes` monotonically non-decreasing in tier index
- K largest tier-1 deltas fit within `escalation_ram_pool_bytes`

If any invariant fails, the `PlanBundle` constructor raises with a specific message. A plan that constructs successfully is guaranteed not to exceed the memory budget under the escalation policy's `max_concurrent_escalations` bound.

### Runtime

```
Executor.step_token(hidden)
    │
    ├─ for each op_bundle:
    │      TierController.active_op(op_id)   ← O(1) lookup
    │      PrefetchScheduler.on_op_start(op) ← issue SSD reads
    │      PrefetchScheduler.wait_until_resident(requires, timeout)
    │      FallbackPlanner.decide(op, missed) ← deadline-miss handling
    │      OpKernel.execute(op, decision, hidden)
    │      RuntimeMonitor.record_op(predicted_us, measured_us)
    │      VerifierProbe.disagreement(op_id, hidden)
    │      TierController.observe(op_id, disagreement)
    │      ResidencyManager.apply_evict_rules(op)
    │
    ├─ end-of-token:
    │      TierController.end_token()         ← decay persistence counters
    │      RuntimeMonitor.assess()            ← check for critical drift
    │      → LATENCY_CRITICAL / SSD_BW_CRITICAL: force_demote_all()
    │
    └─ return hidden
```

**`TierController`** — the state machine. It is explicitly not a replanner. It selects among compiled tiers, maintains an escalation pool budget, and decays escalations over time via persistence counters. When the pool is full and a new escalation is needed, it evicts the laziest currently-escalated op (lowest recent disagreement, most persistence remaining).

**`RuntimeMonitor`** — sliding-window measurement of actual vs predicted latency and SSD bandwidth. Raises `LATENCY_CRITICAL` or `SSD_BW_CRITICAL` signals that trigger `force_demote_all`.

**`VerifierProbe`** — the disagreement sensor. v0.1 ships a `LinearProbeVerifier` (per-layer linear probe over hidden states, trained offline) and a `ScriptedVerifier` for testing. A companion-model verifier and self-consistency verifier are planned.

**`PrefetchScheduler`** — EDF (Earliest-Deadline-First) scheduler with sliding-window bandwidth admission control. Issues SSD reads on a worker thread during the preceding op's compute window. Missed deadlines trigger the op's configured `FallbackStrategy`.

---

## The Rate-Distortion Framing

Substrate treats precision and residency allocation as a rate-distortion problem:

```
rate        = memory consumption + SSD bandwidth + precision bits used
distortion  = quality loss + verifier disagreement + KL drift from reference
```

The compiler solves:

```
minimize   distortion
subject to rate ≤ budget
```

The greedy RDO fill iteratively promotes the operation with the highest `(Δquality / ΔRAM) × (1 + sensitivity)` ratio until either the RAM budget is exhausted or the quality cap is met. This is a known-good approximation for single-machine job scheduling with heterogeneous costs; the MoE structure makes it particularly effective because expert layers have large Δquality per byte at low precision.

This gives the project mathematical legitimacy. It is not "stream things from disk and hope." It is a constrained optimization problem with a defined objective and a solver that runs offline.

---

## Rate-Distortion vs Static Quantization

| | Static Quantization | Substrate |
|---|---|---|
| Precision decision | Once, at load time | Per-op, at compile time |
| Runtime adaptation | None | Tier switching on disagreement signal |
| RAM guarantee | Yes (fixed) | Yes (escalation pool bounds it) |
| Quality under hard prompts | Uniform degradation | Targeted escalation on sensitive layers |
| Killer use case | Dense models | MoE models with cold expert pools |
| Planning cost | Negligible | Offline, once per model+budget pair |
| Latency overhead | None | O(1) tier switch + prefetch window |

The MoE case is where Substrate's advantage is clearest. In a 64-expert model with top-k=8, 56 experts are cold on every token. Static quantization pays full precision for all 64. Substrate keeps frequently-activated experts at higher precision and streams cold ones from SSD, with the tier ladder providing a validated quality floor at every point on the latency-quality tradeoff curve.

---

## Project Status

**v0.1.0 — scaffold complete, MLX integration pending**

### What's Built (and Tested)

| Component | Status |
|---|---|
| `compiler/ir.py` — PlanBundle IR with invariant validation | ✅ Complete |
| `compiler/planner.py` — greedy RDO → PlanBundle | ✅ Complete |
| `compiler/feasibility.py` — 4-axis check with relax_options | ✅ Complete |
| `compiler/quality.py` — HybridQualityEstimator | ✅ Complete |
| `runtime/tier_controller.py` — tier state machine | ✅ Complete |
| `runtime/executor.py` — token loop | ✅ Complete |
| `runtime/monitor.py` — drift detection | ✅ Complete |
| `runtime/verifier.py` — LinearProbeVerifier + ScriptedVerifier | ✅ Complete |
| `runtime/prefetch.py` — EDF prefetch scheduler | ✅ Complete |
| `runtime/residency.py` — RAM budget enforcement | ✅ Complete |
| `runtime/fallback.py` — deadline-miss handler | ✅ Complete |
| `models/quantized_store.py` — on-disk format | ✅ Complete |
| `models/loaders.py` — profile loader + synthetic generators | ✅ Complete |
| `models/moe.py` — MoE shared-expert provider + expert history | ✅ Complete |
| `traces/schema.py` + `recorder.py` — JSONL trace system | ✅ Complete |
| `cli.py` — compile / inspect / feasibility / synth-moe | ✅ Complete |
| Calibration tooling (`scripts/calibrate.py`) | 🔲 Next |
| MLX kernel implementation (`OpKernel` backend) | 🔲 Next |
| Real SSD I/O in `PrefetchScheduler` | 🔲 Next |
| Linear probe training script | 🔲 Next |
| v0.2: residency-aware tier escalation | 🔲 Planned |
| v0.3: adaptive replanning, learned residency policy | 🔲 Planned |

### Known Limitations in v0.1

- **`predicted_tokens_per_second`** uses linear interpolation between `skeleton_compute_us` and `full_precision_compute_us`. Replace with measured values once calibration runs are available.
- **Quality aggregation** is additive (upper bound). Errors compound non-linearly through transformer layers; a non-linear aggregator trained on real calibration data will be more accurate.
- **`QualityEstimator` calibration table** is populated by `stub_calibration_table()` until a real calibration run is performed. Any plan compiled against stub data produces fictional quality estimates.
- **KV budget** is not jointly solved with weight residency. v0.1 treats KV as a fixed overhead.
- **SSD I/O** in `PrefetchScheduler._service_job` uses `time.sleep(bytes / bandwidth)` to simulate read latency. Real I/O (`mmap`, async reads) is the next integration point.

---

## Build Order

1. **IR + validation** ✅ — `compiler/ir.py`. If the IR is wrong, calibration work becomes expensive trash. Build the skeleton before feeding it real measurements.
2. **Static fake profile + planner smoke test** ✅ — `synthesize_moe_profile()` + `Planner.compile()`.
3. **Calibration tooling** 🔲 — Offline runner that produces `(layer × op_kind × precision → KL/perplexity delta)` tables. The `HybridQualityEstimator` consumes these.
4. **MLX executor** 🔲 — Implement `OpKernel` against MLX arrays. The protocol is defined; the implementation is the next chunk.
5. **SSD streamer** 🔲 — Wire real I/O into `PrefetchScheduler`. `mmap` of the `QuantizedStore` tensors is the natural path on macOS/Apple Silicon.
6. **Verifier/tier controller** ✅ — Interface + `LinearProbeVerifier` stub + `ScriptedVerifier` for testing. Probe training requires calibration data first.
7. **ANE/AMX optimization** 🔲 — v0.2+.

---

## Installation

```bash
git clone https://github.com/Jbase16/Substrate.git
cd Substrate
pip install -e .
```

Requires Python 3.11+. No dependencies beyond the standard library in v0.1. MLX will be a dependency once the kernel backend is implemented.

---

## CLI

```bash
# Synthesize a test MoE calibration profile (no real model needed)
substrate synth-moe --layers 27 --experts 64 --top-k 8 --out moe_calib.json

# Compile a plan under a 36GB RAM budget
substrate compile moe_calib.json \
  --max-ram 36G \
  --quality 0.05 \
  --tps 4 \
  --max-concurrent 6 \
  --out plan.json

# Inspect the compiled plan
substrate inspect plan.json

# Check feasibility without compiling
substrate feasibility moe_calib.json --max-ram 2G --quality 0.05
# → {"status": "infeasible", "binding_axis": "memory", "relax_options": {...}}
```

---

## Python API

```python
from substrate import Budget, EscalationPolicy, FallbackPolicy, Planner
from substrate.models.loaders import synthesize_moe_profile
from substrate.runtime.tier_controller import TierController
from substrate.runtime.verifier import ScriptedVerifier

# Build a synthetic MoE profile (replace with real calibration for production)
profile = synthesize_moe_profile(
    model_id="qwen-moe-30b",
    num_layers=27,
    num_experts=64,
    top_k=8,
    sensitive_layer_ids=(0, 1, 25, 26),
)

# Define the hardware budget
budget = Budget(
    max_ram_bytes=int(36e9),
    max_ssd_cache_bytes=int(400e9),
    sustained_ssd_bw_bytes_per_sec=int(5e9),
    quality_loss_cap=0.05,
    target_tokens_per_second=4.0,
)

# Compile
plan = Planner(
    escalation_policy=EscalationPolicy(
        disagreement_threshold=0.15,
        max_concurrent_escalations=6,
        persistence_tokens=32,
    ),
    fallback_policy=FallbackPolicy(),
).compile(profile, budget)

print(f"ops: {plan.num_ops}")
print(f"default path peak RAM: {plan.predicted_peak_resident_bytes / 1e9:.2f} GB")
print(f"escalation pool: {plan.escalation_ram_pool_bytes / 1e6:.0f} MB")

# Inspect a tier ladder
ob = plan.bundle("block_5.attn")
for tier in ob.tiers:
    print(f"  tier {tier.tier_index}: quality_risk={tier.estimated_quality_risk:.4f}, "
          f"Δram={tier.peak_ram_delta_bytes / 1e6:.1f} MB")

# Simulate runtime tier escalation
ctrl = TierController(plan)
verifier = ScriptedVerifier({"block_5.attn": [0.0, 0.0, 0.5, 0.5, 0.5, 0.0]})

for token in range(6):
    for ob in plan.op_bundles:
        disagreement = verifier.disagreement(ob.op_id, hidden=None)
        ctrl.observe(ob.op_id, disagreement)
    ctrl.end_token()
    active = ctrl.active_op("block_5.attn")
    print(f"token {token}: block_5.attn @ tier {active.tier_index}, "
          f"pool_used={ctrl.pool_used_bytes / 1e6:.1f} MB")
```

---

## On-Disk Format

The `QuantizedStore` uses a simple directory layout designed for direct `mmap` access:

```
model_store/
  manifest.json         # Tensor metadata index
  tensors/
    block_0.attn.skeleton.bin
    block_0.attn.residual_1.bin
    block_0.attn.residual_2.bin
    block_0.router.skeleton.bin
    block_0.moe.skeleton.bin
    block_0.moe.residual_1.bin
    ...
```

`manifest.json` maps tensor names to shapes, dtypes, byte counts, and tier indices. The naming convention is `{op_id}.{skeleton|residual_N}`, which matches the IR's `TensorRef` identifiers directly.

---

## Target Demo: MoE Under 36 GB

The headline target for v0.1b is a DeepSeek-V2 / Qwen-MoE class model (27 layers, 64 experts, top-k=8) running under a 36 GB unified memory cap on Apple Silicon.

At that scale, the escalation pool after the default path's peak resident set is ~30 GB. The K=6 largest tier-1 escalations cost ~104 MB total. The system can escalate the 6 most quality-sensitive ops on every token with 104 MB of pool headroom — a rounding error in a 36 GB budget.

The demo is: verifier shows quality drift on attention layers under a hard prompt → tier controller escalates those layers to tier 1 within one token → quality recovers → ops decay back to default after 32 tokens of stability. No pause. No replan. No beachball.

Dense 70B is the debugging target. MoE is the proof.

---

## License

MIT
