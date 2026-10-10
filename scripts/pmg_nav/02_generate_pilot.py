#!/usr/bin/env python3
"""Generate deterministic, oracle-labeled Stage 2 pilot instructions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import sys

PROJECT_MAIN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_MAIN))

from pmg_nav import ConstraintCompiler, Evidence, PhysicalGoalField, PhysicalMemoryGraph


STOPWORDS = {"root", "hospital", "building", "room", "floor", "wall", "ceiling", "mesh", "object", "group", "default", "material", "instance"}
TEMPLATES = (
    "Stop {distance:.1f} meters right of the {anchor} and remain reachable.",
    "Stop {distance:.1f} meters left of the {anchor} and remain reachable.",
    "Stop {distance:.1f} meters in front of the {anchor} and remain reachable.",
    "Stop {distance:.1f} meters behind the {anchor} and remain reachable.",
    "Navigate near the {anchor} and do not block the passage.",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=36)
    return parser.parse_args()


def select_anchors(objects: list[dict]) -> list[dict]:
    candidates = []
    for item in objects:
        label = item["label"]
        tokens = set(label.split())
        extent = item.get("extent", [0, 0, 0])
        if tokens.intersection(STOPWORDS) or len(label) < 3:
            continue
        if not all(math.isfinite(float(value)) for value in item["position"]):
            continue
        if max(extent, default=0) < 0.15 or max(extent, default=0) > 15:
            continue
        candidates.append(item)
    label_counts = Counter(item["label"] for item in candidates)
    unique = [item for item in candidates if label_counts[item["label"]] == 1]
    return unique or candidates


def main() -> None:
    args = parse_args()
    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    anchors = select_anchors(payload["objects"])
    if len(anchors) < 5:
        raise RuntimeError(f"annotation gate found only {len(anchors)} usable anchors")
    rng = random.Random(args.seed)
    compiler = ConstraintCompiler()
    goal_field = PhysicalGoalField()
    records = []
    for index in range(args.count):
        anchor = anchors[index % len(anchors)]
        distance = (1.0, 1.5, 2.0, 2.5, 3.0)[(index // len(anchors)) % 5]
        template = TEMPLATES[(index // (len(anchors) * 5)) % len(TEMPLATES)]
        instruction = template.format(distance=distance, anchor=anchor["label"])
        if index >= len(anchors) * len(TEMPLATES) * 5:
            suffix = (" Approach carefully.", " Keep a clear path.", " Use the nearest valid pose.")[rng.randrange(3)]
            instruction += suffix
        graph = PhysicalMemoryGraph()
        node = graph.observe(anchor["label"], tuple(anchor["position"]), Evidence("simulator_oracle", confidence=1.0), node_id=anchor["prim_path"])
        program = compiler.compile(instruction)
        goals = goal_field.sample_and_rank(graph, program, top_k=3)
        records.append(
            {
                "id": f"pilot_{index:04d}",
                "instruction": instruction,
                "constraint_program": program.to_dict(),
                "anchor": node.to_dict(),
                "oracle_goal_poses": [vars(goal) for goal in goals],
                "unique_solution": goals[0].score - goals[1].score > 0.05 if len(goals) > 1 else True,
            }
        )
    if len({record["instruction"] for record in records}) != len(records):
        raise AssertionError("pilot instructions are not unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps({"count": len(records), "usable_anchors": len(anchors), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
