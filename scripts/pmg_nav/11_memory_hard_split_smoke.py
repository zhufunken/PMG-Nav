#!/usr/bin/env python3
"""Generate and evaluate a tiny deterministic memory-dependent PMG fixture.

This is an offline contract smoke test.  It does not start Isaac Sim, load
TIC-VLA, render images, or claim navigation performance.  The fixture makes
the history dependency explicit so it can be validated before adding any
DynaNav scene/episode hooks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any


PROJECT_MAIN = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_MAIN / "outputs/stage2/annotation/memory_hard_split_smoke.jsonl"
DEFAULT_REPORT = PROJECT_MAIN / "outputs/stage2/annotation/memory_hard_split_smoke_report.json"
DEFAULT_SEED = 20260903


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _checksum(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _position(value: tuple[float, float, float]) -> list[float]:
    return [round(float(item), 6) for item in value]


def make_episode(index: int, *, seed: int) -> dict[str, Any]:
    """Build one paired fixture episode without any random global state."""

    episode_seed = seed + index
    split = "train" if index < 3 else "test"
    x = 4.0 + index * 0.5
    anchor_a = (x, 0.0, 0.0)
    anchor_b = (x + 2.0, 0.0, 0.0)
    target_first = (x + 2.0, 2.0, 0.0)
    target_final = (x + 2.5, 2.0, 0.0)
    distractor_one = (x - 1.5, 2.0, 0.0)
    distractor_two = (x - 1.5, 2.7, 0.0)
    goal = (target_final[0] + 0.5, target_final[1], target_final[2])

    if split == "train":
        relation_types = ["left_of", "facing"]
        metric_type = "near_distance"
        hops = [
            ["anchor_a", "left_of", "anchor_b"],
            ["anchor_b", "facing", "target"],
        ]
    else:
        relation_types = ["between", "behind"]
        metric_type = "metric_distance"
        hops = [
            ["anchor_a", "between", "anchor_b"],
            ["anchor_b", "behind", "target"],
        ]

    return {
        "id": f"memory_{split}_{index:03d}",
        "split": split,
        "episode_seed": episode_seed,
        "instruction": (
            "Use the anchor_a to anchor_b relation, verify the reappearing target, "
            "then stop 0.5 meters right of the target."
        ),
        "relation_program": {
            "hops": hops,
            "relation_types": relation_types,
            "metric": {"anchor": "target", "relation": metric_type, "value": 0.5},
            "history_required": True,
        },
        "objects": {
            "anchor_a": _position(anchor_a),
            "anchor_b": _position(anchor_b),
            "target": _position(target_final),
            "target_distractor_1": _position(distractor_one),
            "target_distractor_2": _position(distractor_two),
        },
        "history": [
            {
                "frame": 0,
                "visible": [
                    {"id": "anchor_a", "label": "anchor", "position": _position(anchor_a), "confidence": 1.0},
                    {"id": "anchor_b", "label": "anchor", "position": _position(anchor_b), "confidence": 1.0},
                    {"id": "target", "label": "target", "position": _position(target_first), "confidence": 0.95},
                ],
                "occluded": [],
                "contradictions": [],
            },
            {
                "frame": 1,
                "visible": [
                    {"id": "anchor_a", "label": "anchor", "position": _position(anchor_a), "confidence": 0.9},
                ],
                "occluded": ["anchor_b", "target"],
                "contradictions": [
                    {"id": "target", "position": _position((x + 7.0, -3.0, 0.0)), "confidence": 0.2}
                ],
            },
            {
                "frame": 2,
                "visible": [
                    {"id": "anchor_b", "label": "anchor", "position": _position(anchor_b), "confidence": 0.95},
                    {"id": "target_distractor_1", "label": "target", "position": _position(distractor_one), "confidence": 0.82},
                    {"id": "target_distractor_2", "label": "target", "position": _position(distractor_two), "confidence": 0.80},
                    {"id": "target", "label": "target", "position": _position(target_final), "confidence": 0.88},
                ],
                "occluded": [],
                "reappeared": ["anchor_b", "target"],
                "pose_changed": ["target"],
                "contradictions": [],
            },
        ],
        "oracle_goal_pose": _position(goal),
        "requirements": {
            "multi_hop_relation": 2,
            "similar_object_distractors": 2,
            "anchor_occluded_then_reappears": True,
            "target_pose_changes_after_first_observation": True,
            "contradictory_observation": True,
            "revisit_or_verification_required": True,
            "history_is_necessary_for_success": True,
        },
        "verification": {"required": True, "revisit_anchor": "anchor_b", "contradiction_policy": "retain_consistent_track"},
    }


def write_fixture(path: Path, *, episodes: int = 6, seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    if episodes < 4:
        raise ValueError("episodes must be at least 4 so train/test relation families are both represented")
    rows = [make_episode(index, seed=seed) for index in range(episodes)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return rows


def _validate_fixture(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("fixture is empty")
    train = [row for row in rows if row.get("split") == "train"]
    test = [row for row in rows if row.get("split") == "test"]
    if not train or not test:
        raise ValueError("fixture must contain both train and test rows")
    for row in rows:
        requirements = row.get("requirements")
        if not isinstance(requirements, dict) or not all(requirements.values()):
            raise ValueError(f"hard-split requirements are incomplete for {row.get('id')}")
        if int(requirements.get("multi_hop_relation", 0)) < 2:
            raise ValueError(f"multi-hop relation depth is too small for {row.get('id')}")
        if int(requirements.get("similar_object_distractors", 0)) < 2:
            raise ValueError(f"similar-object distractor count is too small for {row.get('id')}")
        history = row.get("history")
        if not isinstance(history, list) or len(history) < 3:
            raise ValueError(f"history is incomplete for {row.get('id')}")
        if len(row.get("relation_program", {}).get("hops", [])) < 2:
            raise ValueError(f"two-hop relation is missing for {row.get('id')}")
    train_types = {item for row in train for item in row["relation_program"]["relation_types"]}
    test_types = {item for row in test for item in row["relation_program"]["relation_types"]}
    if train_types & test_types:
        raise ValueError("train/test relation families overlap")
    train_metrics = {row["relation_program"]["metric"]["relation"] for row in train}
    test_metrics = {row["relation_program"]["metric"]["relation"] for row in test}
    if train_metrics & test_metrics:
        raise ValueError("train/test metric families overlap")
    return {
        "rows": len(rows),
        "train_rows": len(train),
        "test_rows": len(test),
        "history_required": True,
        "train_relation_types": sorted(train_types),
        "test_relation_types": sorted(test_types),
        "train_metric_types": sorted(train_metrics),
        "test_metric_types": sorted(test_metrics),
        "no_relation_metric_pair_overlap": True,
    }


def evaluate_fixture(rows: list[dict[str, Any]], *, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    contract = _validate_fixture(rows)
    methods: dict[str, list[dict[str, Any]]] = {name: [] for name in ("M0", "M1", "M2", "M3")}
    for index, row in enumerate(rows):
        objects = row["objects"]
        goal = tuple(float(value) for value in row["oracle_goal_pose"])
        distractor = tuple(float(value) for value in objects["target_distractor_1"])
        # M0 sees only the final ambiguous frame and deterministically picks the
        # first same-label distractor. M1 uses the stable identity across history.
        m0 = {"x": distractor[0] + 0.5, "y": distractor[1], "yaw": 0.0}
        m1 = {"x": goal[0], "y": goal[1], "yaw": 0.0}
        episode_rng = random.Random(seed + index)
        noise = episode_rng.uniform(-0.25, 0.25)
        m2 = {"x": goal[0] + noise, "y": goal[1] + noise, "yaw": 0.0}
        actions = {"M0": m0, "M1": m1, "M2": m2, "M3": m0}
        for method, action in actions.items():
            error = _distance((action["x"], action["y"], 0.0), goal)
            success = error <= 0.75
            methods[method].append(
                {
                    "episode": row["id"],
                    "goal_pose_error_m": error,
                    "sr": int(success),
                    "spl": 1.0 / (1.0 + error) if success else 0.0,
                    "action": action,
                }
            )

    summary: dict[str, Any] = {}
    for method, values in methods.items():
        summary[method] = {
            "episodes": len(values),
            "sr": statistics.fmean(item["sr"] for item in values),
            "spl": statistics.fmean(item["spl"] for item in values),
            "goal_pose_error_m": statistics.fmean(item["goal_pose_error_m"] for item in values),
            "action_output_checksum": _checksum([item["action"] for item in values]),
        }

    paired: dict[str, Any] = {}
    for method in ("M1", "M2", "M3"):
        deltas = [
            new["goal_pose_error_m"] - old["goal_pose_error_m"]
            for old, new in zip(methods["M0"], methods[method])
        ]
        paired[method] = {
            "episodes": len(deltas),
            "goal_pose_error_delta_m": statistics.fmean(deltas),
            "sr_delta": statistics.fmean(new["sr"] - old["sr"] for old, new in zip(methods["M0"], methods[method])),
            "action_checksum_equal_m0": summary[method]["action_output_checksum"] == summary["M0"]["action_output_checksum"],
        }

    return {
        "schema_version": 1,
        "type": "memory_dependent_hard_split_offline_smoke",
        "fixture_only": True,
        "strict_end_to_end": False,
        "isaac_started": False,
        "model_loaded": False,
        "pmg_runtime_used": False,
        "method_implementation": {
            "M0": "current_frame_ambiguous_baseline",
            "M1": "history_stable_track_oracle",
            "M2": "history_stable_track_noisy_oracle",
            "M3": "M0_action_bypass",
        },
        "seed": seed,
        "contract": contract,
        "methods": summary,
        "paired_deltas_vs_m0": paired,
        "claims_not_proven": [
            "Isaac Sim or DynaNav navigation success",
            "TIC-VLA action quality or closed-loop SR/SPL",
            "PMG runtime/goal-field/action-adapter integration",
            "perception, depth projection, or object association accuracy",
            "generalization beyond this synthetic fixture",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    rows = write_fixture(args.output.resolve(), episodes=args.episodes, seed=args.seed)
    report = evaluate_fixture(rows, seed=args.seed)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
