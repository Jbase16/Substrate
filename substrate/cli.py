"""
substrate.cli — Command-line entrypoint.

Commands:
    substrate compile <calibration.json> --max-ram 36G --quality 0.05 --out plan.json
    substrate inspect <plan.json>
    substrate feasibility <calibration.json> --max-ram 36G ...
    substrate synth-moe --layers 27 --experts 64 --out moe.json
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
from substrate.compiler.feasibility import InfeasibleBudgetError, check_feasibility
from substrate.compiler.ir import Budget, EscalationPolicy, FallbackPolicy, PlanBundle
from substrate.compiler.planner import Planner, PlannerOptions
from substrate.models.loaders import load_profile_from_calibration, synthesize_moe_profile


def _parse_size(s: str) -> int:
    s = s.strip().upper()
    mul = {"K": 1_000, "M": 1_000_000, "G": 1_000_000_000, "T": 1_000_000_000_000}
    return int(float(s[:-1]) * mul[s[-1]]) if s and s[-1] in mul else int(s)


def _make_budget(args) -> Budget:
    return Budget(
        max_ram_bytes=_parse_size(args.max_ram),
        max_ssd_cache_bytes=_parse_size(args.ssd_cache),
        sustained_ssd_bw_bytes_per_sec=_parse_size(args.ssd_bw),
        quality_loss_cap=args.quality,
        target_tokens_per_second=args.tps,
    )


def cmd_compile(args) -> int:
    profile = load_profile_from_calibration(args.calibration)
    budget = _make_budget(args)
    planner = Planner(
        opts=PlannerOptions(emit_escalation_tiers=not args.no_escalation),
        escalation_policy=EscalationPolicy(max_concurrent_escalations=args.max_concurrent),
        fallback_policy=FallbackPolicy(),
    )
    try:
        plan = planner.compile(profile, budget)
    except InfeasibleBudgetError as e:
        print("INFEASIBLE:", e.report.reason, file=sys.stderr)
        print(json.dumps(e.report.to_dict(), indent=2), file=sys.stderr)
        return 2
    _write_plan_summary(plan, Path(args.out))
    print(f"Compiled plan -> {args.out}")
    print(f"  ops: {plan.num_ops}")
    print(f"  predicted RAM: {plan.predicted_peak_resident_bytes / 1e9:.2f} GB / {budget.max_ram_bytes / 1e9:.2f} GB cap")
    print(f"  predicted quality loss: {plan.predicted_quality_loss:.4f} / {budget.quality_loss_cap:.4f} cap")
    print(f"  escalation pool: {plan.escalation_ram_pool_bytes / 1e6:.1f} MB")
    print(f"  predicted tok/s: {plan.predicted_tokens_per_second:.2f}")
    return 0


def cmd_inspect(args) -> int:
    with open(args.plan) as f:
        data = json.load(f)
    print(f"Plan: {data['model_id']}  (solver: {data['solver_version']})")
    print(f"  ops: {len(data['op_bundles'])}")
    print(f"  predicted_peak_ram: {data['predicted_peak_resident_bytes'] / 1e9:.2f} GB")
    print(f"  pool: {data['escalation_ram_pool_bytes'] / 1e6:.1f} MB")
    print(f"  predicted_quality_loss: {data['predicted_quality_loss']:.4f}")
    print(f"  predicted_tps: {data['predicted_tokens_per_second']:.2f}")
    tier_counts = {}
    for ob in data["op_bundles"]:
        n = len(ob["tiers"])
        tier_counts[n] = tier_counts.get(n, 0) + 1
    print("\nTier distribution:")
    for n, c in sorted(tier_counts.items()):
        print(f"  ops with {n} tier(s): {c}")
    print("\nSolver notes:")
    for note in data.get("solver_notes", []):
        print(f"  - {note}")
    return 0


def cmd_feasibility(args) -> int:
    report = check_feasibility(load_profile_from_calibration(args.calibration), _make_budget(args))
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.feasible else 2


def cmd_synth_moe(args) -> int:
    profile = synthesize_moe_profile(
        model_id=args.model_id, num_layers=args.layers, num_experts=args.experts,
        top_k=args.top_k, hidden_size=args.hidden,
    )
    data = {
        "model_id": profile.model_id,
        "embedding_bytes": profile.embedding_bytes,
        "lm_head_bytes": profile.lm_head_bytes,
        "runtime_overhead_bytes": profile.runtime_overhead_bytes,
        "ops": [{"op_id": op.op_id, "op_kind": op.op_kind.value, "layer_id": op.layer_id,
                 "param_count": op.param_count, "skeleton_compute_us": op.skeleton_compute_us,
                 "full_precision_compute_us": op.full_precision_compute_us, "sensitivity": op.sensitivity,
                 "minimum_residual_bytes_per_token": op.minimum_residual_bytes_per_token,
                 "moe_top_k": op.moe_top_k, "moe_num_experts": op.moe_num_experts} for op in profile.ops],
    }
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote synthetic MoE profile -> {args.out}")
    print(f"  layers: {args.layers}  experts: {args.experts}  top_k: {args.top_k}")
    return 0


def _write_plan_summary(plan: PlanBundle, path: Path):
    data = {
        "model_id": plan.model_id, "solver_version": plan.solver_version,
        "predicted_peak_resident_bytes": plan.predicted_peak_resident_bytes,
        "predicted_steady_state_resident_bytes": plan.predicted_steady_state_resident_bytes,
        "predicted_ssd_bandwidth_bps": plan.predicted_ssd_bandwidth_bps,
        "predicted_tokens_per_second": plan.predicted_tokens_per_second,
        "predicted_quality_loss": plan.predicted_quality_loss,
        "escalation_ram_pool_bytes": plan.escalation_ram_pool_bytes,
        "solver_notes": list(plan.solver_notes),
        "budget": {"max_ram_bytes": plan.budget.max_ram_bytes, "max_ssd_cache_bytes": plan.budget.max_ssd_cache_bytes,
                   "sustained_ssd_bw_bytes_per_sec": plan.budget.sustained_ssd_bw_bytes_per_sec,
                   "quality_loss_cap": plan.budget.quality_loss_cap, "target_tokens_per_second": plan.budget.target_tokens_per_second},
        "escalation_policy": {"disagreement_threshold": plan.escalation_policy.disagreement_threshold,
                              "consecutive_hits_for_tier_2": plan.escalation_policy.consecutive_hits_for_tier_2,
                              "persistence_tokens": plan.escalation_policy.persistence_tokens,
                              "max_concurrent_escalations": plan.escalation_policy.max_concurrent_escalations,
                              "enable_demotion": plan.escalation_policy.enable_demotion},
        "fallback_policy": {"deadline_miss_strategy": plan.fallback_policy.deadline_miss_strategy.value,
                            "critical_latency_factor": plan.fallback_policy.critical_latency_factor,
                            "critical_ssd_bw_factor": plan.fallback_policy.critical_ssd_bw_factor},
        "tensor_catalog": [{"tensor_id": m.tensor_id, "bytes_in_ram": m.bytes_in_ram, "bytes_on_ssd": m.bytes_on_ssd,
                             "is_skeleton": m.is_skeleton, "layer_id": m.layer_id, "tier_index": m.tier_index}
                            for m in plan.tensor_catalog.values()],
        "op_bundles": [{"op_id": ob.op_id, "tiers": [
            {"tier_index": t.tier_index, "op_kind": t.op_kind.value, "layer_id": t.layer_id,
             "requires": [r.tensor_id for r in t.requires],
             "prefetch": [{"tensor": p.tensor.tensor_id, "start_during": p.start_during,
                           "deadline_before": p.deadline_before, "priority": p.priority} for p in t.prefetch],
             "evict_after": [{"tensor": e.tensor.tensor_id, "after_op": e.after_op} for e in t.evict_after],
             "fallback": t.fallback.value, "estimated_compute_us": t.estimated_compute_us,
             "estimated_quality_risk": t.estimated_quality_risk, "peak_ram_delta_bytes": t.peak_ram_delta_bytes,
             "moe_likely_experts": list(t.moe_likely_experts), "moe_top_k": t.moe_top_k}
            for t in ob.tiers]} for ob in plan.op_bundles],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _build_parser():
    p = argparse.ArgumentParser(prog="substrate", description="Substrate — MLX-native execution planner.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_budget(sp):
        sp.add_argument("--max-ram", required=True)
        sp.add_argument("--ssd-cache", default="400G")
        sp.add_argument("--ssd-bw", default="5G")
        sp.add_argument("--quality", type=float, default=0.05)
        sp.add_argument("--tps", type=float, default=None)

    pc = sub.add_parser("compile")
    pc.add_argument("calibration")
    pc.add_argument("--out", required=True)
    pc.add_argument("--no-escalation", action="store_true")
    pc.add_argument("--max-concurrent", type=int, default=8)
    add_budget(pc)

    pi = sub.add_parser("inspect")
    pi.add_argument("plan")

    pf = sub.add_parser("feasibility")
    pf.add_argument("calibration")
    add_budget(pf)

    ps = sub.add_parser("synth-moe")
    ps.add_argument("--out", required=True)
    ps.add_argument("--model-id", default="synth-moe")
    ps.add_argument("--layers", type=int, default=27)
    ps.add_argument("--experts", type=int, default=64)
    ps.add_argument("--top-k", type=int, default=8)
    ps.add_argument("--hidden", type=int, default=4096)
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    return {"compile": cmd_compile, "inspect": cmd_inspect, "feasibility": cmd_feasibility, "synth-moe": cmd_synth_moe}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
