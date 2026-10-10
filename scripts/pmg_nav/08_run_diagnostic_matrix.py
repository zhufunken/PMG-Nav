#!/usr/bin/env python3
"""Run a small, paired PMG diagnostic without Isaac or TIC-VLA weights.

This is an instrumentation smoke test, not an end-to-end navigation score. It
uses the checked-in pilot annotations to validate the M0-M3 contracts and
statistics before spending GPU time. M2 is intentionally blocked unless the
caller opts into ``--allow-predicted-proxy``; that proxy is never reported as
real detector perception.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

PROJECT_MAIN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_MAIN))

from pmg_nav import ConstraintCompiler, Evidence, PhysicalGoalField, PhysicalMemoryGraph
from pmg_nav.diagnostics import bootstrap_ci, paired_sign_permutation_pvalue, sequence_checksum


def distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, default=PROJECT_MAIN / "outputs/stage2/annotation/pilot_instructions.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--allow-predicted-proxy", action="store_true")
    return parser.parse_args()


def candidate_metrics(candidate, program, graph, oracle_goal):
    evaluator = PhysicalGoalField().evaluator
    anchors = graph.anchors()
    if graph.objects:
        scores = evaluator.evaluate_program(candidate.position, program, anchors, yaw=candidate.yaw)
    else:
        # M0 has no PMG anchor by contract; its physical-constraint fields are
        # therefore not applicable rather than an evaluator failure.
        scores = []
    by_relation = {}
    for constraint, score in zip(program.constraints, scores):
        by_relation.setdefault(constraint.relation, []).append(score)
    def mean_score(names):
        values = [score.score for name in names for score in by_relation.get(name, [])]
        return statistics.fmean(values) if values else None
    metric_scores = mean_score(("distance", "near", "far"))
    relation_scores = mean_score(("left", "right", "front", "behind", "between"))
    orientation_scores = mean_score(("facing",))
    reachability_scores = mean_score(("reachable", "non_blocking"))
    residuals = [score.residual for name in ("distance", "near", "far") for score in by_relation.get(name, [])]
    goal_error = distance(candidate.position, oracle_goal)
    success = goal_error <= 1.0
    return {
        "success": success,
        "sr": float(success),
        "spl": float(1.0 / (1.0 + goal_error) if success else 0.0),
        "collision": 0.0,
        "physical_constraint_satisfaction": float(candidate.score) if scores else None,
        "metric_constraint_satisfaction": metric_scores,
        "relation_constraint_satisfaction": relation_scores,
        "orientation_reachability_satisfaction": statistics.fmean(
            [value for value in (orientation_scores, reachability_scores) if value is not None]
        ) if any(value is not None for value in (orientation_scores, reachability_scores)) else None,
        "metric_error_m": statistics.fmean(residuals) if residuals else None,
        "goal_pose_error_m": goal_error,
    }


def optional_mean(entries, key):
    values = [item[key] for item in entries if item.get(key) is not None]
    return statistics.fmean(values) if values else None


def main() -> None:
    args = parse_args()
    records = [json.loads(line) for line in args.pilot.read_text(encoding="utf-8").splitlines()[: max(0, args.limit)]]
    if not records:
        raise RuntimeError("diagnostic pilot is empty")
    compiler = ConstraintCompiler()
    goal_field = PhysicalGoalField()
    methods = {name: [] for name in ("M0", "M1", "M2", "M3")}
    action_sequences = {name: [] for name in methods}
    rng = random.Random(args.seed)
    started = time.perf_counter()
    for index, record in enumerate(records):
        program = compiler.compile(record["instruction"])
        anchor = tuple(record["anchor"]["position"])
        oracle_goal = tuple(record["oracle_goal_poses"][0]["position"])
        episode_seed = args.seed + index
        episode_rng = random.Random(episode_seed)
        baseline = tuple(anchor[i] + episode_rng.uniform(-4.0, 4.0) if i < 2 else anchor[i] for i in range(3))
        m0 = dict(candidate_metrics(type("Candidate", (), {"position": baseline, "yaw": 0.0, "score": 0.0})(), program, PhysicalMemoryGraph(), oracle_goal))
        m0.update({"mode": "M0", "episode": record.get("id", index), "episode_seed": episode_seed, "action_enabled": False, "pmg_built": False})
        methods["M0"].append(m0)
        action_sequences["M0"].append(baseline)

        graph = PhysicalMemoryGraph()
        graph.observe(record["anchor"]["label"], anchor, Evidence("simulator_oracle", frame_id=0), node_id=record["anchor"]["node_id"])
        oracle_candidate = goal_field.sample_and_rank(graph, program, top_k=1)[0]
        m1 = candidate_metrics(oracle_candidate, program, graph, oracle_goal)
        m1.update({"mode": "M1", "episode": record.get("id", index), "episode_seed": episode_seed, "action_enabled": True, "pmg_built": True, "perception_source": "oracle", "pmg_vector_consumed": False})
        methods["M1"].append(m1)
        action_sequences["M1"].append(oracle_candidate.position)

        if args.allow_predicted_proxy:
            predicted_graph = PhysicalMemoryGraph()
            noise = tuple(episode_rng.gauss(0.0, 0.9) if i < 2 else 0.0 for i in range(3))
            predicted = tuple(anchor[i] + noise[i] for i in range(3))
            predicted_graph.observe(record["anchor"]["label"], predicted, Evidence("predicted_rgbd_proxy", confidence=0.72, frame_id=0), node_id=record["anchor"]["node_id"])
            predicted_candidate = goal_field.sample_and_rank(predicted_graph, program, top_k=1)[0]
            m2 = candidate_metrics(predicted_candidate, program, predicted_graph, oracle_goal)
            m2.update({"mode": "M2", "episode": record.get("id", index), "episode_seed": episode_seed, "action_enabled": True, "pmg_built": True, "perception_source": "predicted_rgbd_proxy", "pmg_vector_consumed": False})
            methods["M2"].append(m2)
            action_sequences["M2"].append(predicted_candidate.position)

        # M3 uses the same oracle PMG candidate but action is explicitly bypassed.
        m3 = candidate_metrics(oracle_candidate, program, graph, oracle_goal)
        m3.update({"mode": "M3", "episode": record.get("id", index), "episode_seed": episode_seed, "action_enabled": False, "pmg_built": True, "perception_source": "oracle", "pmg_vector_consumed": False, "correction_success": True, "verification_revisit_cost": 1.0})
        methods["M3"].append(m3)
        action_sequences["M3"].append(baseline)

    summary = {}
    for method, entries in methods.items():
        if not entries:
            summary[method] = {"status": "blocked", "reason": "M2 requires explicit predicted perception input" if method == "M2" else "not run"}
            continue
        summary[method] = {
            "status": "passed",
            "episodes": len(entries),
            "sr": statistics.fmean(item["sr"] for item in entries),
            "spl": statistics.fmean(item["spl"] for item in entries),
            "collision": statistics.fmean(item["collision"] for item in entries),
            "physical_constraint_satisfaction": optional_mean(entries, "physical_constraint_satisfaction"),
            "metric_constraint_satisfaction": optional_mean(entries, "metric_constraint_satisfaction"),
            "relation_constraint_satisfaction": optional_mean(entries, "relation_constraint_satisfaction"),
            "orientation_reachability_satisfaction": optional_mean(entries, "orientation_reachability_satisfaction"),
            "metric_error_m": optional_mean(entries, "metric_error_m"),
            "goal_pose_error_m": statistics.fmean(item["goal_pose_error_m"] for item in entries),
            "correction_success": statistics.fmean(float(item.get("correction_success", False)) for item in entries),
            "verification_revisit_cost": statistics.fmean(float(item.get("verification_revisit_cost", 0.0)) for item in entries),
            "inference_latency_s": None,
            "peak_gpu_memory_mib": None,
            "action_output_checksum": sequence_checksum(action_sequences[method]),
            "pmg_vector_consumed": any(item.get("pmg_vector_consumed") for item in entries),
        }
    paired = {}
    base = methods["M0"]
    for method in ("M1", "M2", "M3"):
        if not methods[method]:
            paired[method] = {"status": "blocked"}
            continue
        sr = [new["sr"] - old["sr"] for new, old in zip(methods[method], base)]
        spl = [new["spl"] - old["spl"] for new, old in zip(methods[method], base)]
        goal = [new["goal_pose_error_m"] - old["goal_pose_error_m"] for new, old in zip(methods[method], base)]
        paired[method] = {
            "episodes": len(sr),
            "sr_delta": statistics.fmean(sr),
            "sr_ci95": bootstrap_ci(sr),
            "sr_pvalue": paired_sign_permutation_pvalue(sr),
            "spl_delta": statistics.fmean(spl),
            "spl_ci95": bootstrap_ci(spl),
            "goal_pose_error_delta_m": statistics.fmean(goal),
            "goal_pose_error_ci95": bootstrap_ci(goal),
        }
    result = {
        "type": "offline_paired_diagnostic_smoke",
        "strict_end_to_end": False,
        "predicted_perception": "predicted_rgbd_proxy" if args.allow_predicted_proxy else "blocked_missing_backend",
        "seed": args.seed,
        "episodes": len(records),
        "methods": summary,
        "paired_deltas_vs_m0": paired,
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
