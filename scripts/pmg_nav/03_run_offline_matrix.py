#!/usr/bin/env python3
"""Run reproducible M0-M3 Stage 2 oracle/noise/correction experiments."""

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

from pmg_nav import ConstraintCompiler, Evidence, HeuristicToolScheduler, PhysicalGoalField, PhysicalMemoryGraph
from pmg_nav.adapter import build_pmg_context


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--limit", type=int, default=1000)
    return parser.parse_args()


def distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    compiler = ConstraintCompiler()
    goal_field = PhysicalGoalField()
    scheduler = HeuristicToolScheduler()
    methods = {name: [] for name in ("M0", "M1", "M2", "M3")}
    tool_log = []
    with args.pilot.open(encoding="utf-8") as handle:
        records = [json.loads(line) for _, line in zip(range(args.limit), handle)]
    started = time.perf_counter()
    for record in records:
        oracle = tuple(record["anchor"]["position"])
        program = compiler.compile(record["instruction"])
        oracle_goal = tuple(record["oracle_goal_poses"][0]["position"])
        baseline = (oracle[0] + rng.uniform(-4, 4), oracle[1] + rng.uniform(-4, 4), oracle[2])
        methods["M0"].append({"satisfied": distance(baseline, oracle_goal) <= 1.0, "goal_error": distance(baseline, oracle_goal)})
        graph = PhysicalMemoryGraph()
        first_noise = (rng.gauss(0, 0.9), rng.gauss(0, 0.9), 0.0)
        first_position = tuple(oracle[i] + first_noise[i] for i in range(3))
        graph.observe(record["anchor"]["label"], first_position, Evidence("rgbd_projection", confidence=0.72), node_id=record["anchor"]["node_id"])
        stateless_candidates = goal_field.sample_and_rank(graph, program, top_k=3)
        m1_goal = stateless_candidates[0].position
        methods["M1"].append({"satisfied": distance(m1_goal, oracle_goal) <= 1.0, "goal_error": distance(m1_goal, oracle_goal)})
        for frame_id in range(1, 5):
            noise = (rng.gauss(0, 0.9), rng.gauss(0, 0.9), 0.0)
            noisy_position = tuple(oracle[i] + noise[i] for i in range(3))
            graph.observe(
                record["anchor"]["label"],
                noisy_position,
                Evidence("rgbd_projection", confidence=0.72, frame_id=frame_id),
                node_id=record["anchor"]["node_id"],
            )
        candidates = goal_field.sample_and_rank(graph, program, top_k=3)
        m2_goal = candidates[0].position
        methods["M2"].append({"satisfied": distance(m2_goal, oracle_goal) <= 1.0, "goal_error": distance(m2_goal, oracle_goal)})
        conflict_injected = rng.random() < 0.25
        if conflict_injected:
            angle = rng.uniform(0, 2 * math.pi)
            conflicting_position = (oracle[0] + 3.0 * math.cos(angle), oracle[1] + 3.0 * math.sin(angle), oracle[2])
            contradiction = min(1.0, distance(conflicting_position, graph.objects[record["anchor"]["node_id"]].position) / 2.0)
        else:
            contradiction = 0.0
        decision = scheduler.decide(0.92, 0.55, contradiction, candidates[0].hard_valid)
        corrected = decision.tool in {"BACKTRACK_EVIDENCE", "VERIFY_CONSTRAINT", "QUERY_CANDIDATES"}
        if corrected:
            graph.observe(record["anchor"]["label"], oracle, Evidence("simulator_verification", confidence=1.0), node_id=record["anchor"]["node_id"])
            graph.objects[record["anchor"]["node_id"]].position = oracle
            candidates = goal_field.sample_and_rank(graph, program, top_k=3)
        m3_goal = candidates[0].position
        methods["M3"].append({"satisfied": distance(m3_goal, oracle_goal) <= 1.0, "goal_error": distance(m3_goal, oracle_goal)})
        tool_log.append({"id": record["id"], **vars(decision), "corrected": corrected, "conflict_injected": conflict_injected, "contradiction_probability": contradiction})
        if len(tool_log) == 1:
            context = build_pmg_context(record["instruction"], program, graph, candidates)
    summaries = {}
    for name, entries in methods.items():
        summaries[name] = {
            "episodes": len(entries),
            "physical_constraint_satisfaction": sum(item["satisfied"] for item in entries) / len(entries),
            "mean_goal_error_m": statistics.fmean(item["goal_error"] for item in entries),
            "p95_goal_error_m": sorted(item["goal_error"] for item in entries)[int(0.95 * (len(entries) - 1))],
        }
    result = {
        "seed": args.seed,
        "runtime_seconds": time.perf_counter() - started,
        "methods": summaries,
        "tool_calls": len(tool_log),
        "corrections": sum(item["corrected"] for item in tool_log),
        "tool_cost": sum(item["cost"] for item in tool_log),
        "ticvla_pmg_prompt_example": context,
        "tool_log": tool_log,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "tool_log"}, indent=2))


if __name__ == "__main__":
    main()
