#!/usr/bin/env python3
"""Check paired TIC-VLA query observations without requiring action parity.

This is a diagnostic gate for independent Isaac runs.  It compares the
per-query policy payload emitted by the behavior wrapper, including image
content SHA-256 values, robot state/pose, delay, instruction, and frame.
It deliberately does not compare output commands: use ``check_replay_parity``
for the strict M0/M3 causal gate after observation parity is established.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("kind") == "input":
            rows.append(item)
    return rows


def _first_mismatch(left: list[Any], right: list[Any]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _field_mismatches(
    baseline_payloads: list[Any], candidate_payloads: list[Any]
) -> dict[str, int | None]:
    """Return the first query index that differs for each payload field."""
    fields = (
        "instruction",
        "image_refs",
        "image_sha256",
        "robot_state",
        "robot_pose",
        "delay_time",
        "robot_type",
        "inference_step",
    )
    result: dict[str, int | None] = {}
    for field in fields:
        left = [
            payload.get(field) if isinstance(payload, dict) else None
            for payload in baseline_payloads
        ]
        right = [
            payload.get(field) if isinstance(payload, dict) else None
            for payload in candidate_payloads
        ]
        result[field] = _first_mismatch(left, right)
    return result


def check_observations(
    baseline_trace: Path, candidate_trace: Path
) -> dict[str, Any]:
    baseline = _rows(baseline_trace)
    candidate = _rows(candidate_trace)
    baseline_payloads = [item.get("policy_input_payload") for item in baseline]
    candidate_payloads = [item.get("policy_input_payload") for item in candidate]
    payload_mismatch = _first_mismatch(baseline_payloads, candidate_payloads)
    field_mismatches = _field_mismatches(baseline_payloads, candidate_payloads)
    baseline_frames = [item.get("simulation_frame") for item in baseline]
    candidate_frames = [item.get("simulation_frame") for item in candidate]
    baseline_steps = [item.get("inference_step") for item in baseline]
    candidate_steps = [item.get("inference_step") for item in candidate]
    frame_mismatch = _first_mismatch(baseline_frames, candidate_frames)
    step_mismatch = _first_mismatch(baseline_steps, candidate_steps)
    payload_schema_available = bool(
        baseline_payloads
        and candidate_payloads
        and all(
            isinstance(payload, dict)
            and "image_sha256" in payload
            and "robot_state" in payload
            and "robot_pose" in payload
            for payload in baseline_payloads + candidate_payloads
        )
    )
    schema_missing_index = None
    if not payload_schema_available:
        for index, payload in enumerate(baseline_payloads + candidate_payloads):
            if not (
                isinstance(payload, dict)
                and "image_sha256" in payload
                and "robot_state" in payload
                and "robot_pose" in payload
            ):
                schema_missing_index = index
                break
    passed = bool(
        baseline
        and len(baseline) == len(candidate)
        and frame_mismatch is None
        and step_mismatch is None
        and payload_mismatch is None
        and payload_schema_available
    )
    first_payload_baseline = (
        baseline_payloads[payload_mismatch]
        if payload_mismatch is not None and payload_mismatch < len(baseline_payloads)
        else None
    )
    first_payload_candidate = (
        candidate_payloads[payload_mismatch]
        if payload_mismatch is not None and payload_mismatch < len(candidate_payloads)
        else None
    )
    return {
        "schema_version": 1,
        "kind": "stage2_observation_parity_check",
        "baseline_trace": str(baseline_trace),
        "candidate_trace": str(candidate_trace),
        "baseline_input_count": len(baseline),
        "candidate_input_count": len(candidate),
        "observation_payload_schema_available": payload_schema_available,
        "observation_payload_schema_missing_index": schema_missing_index,
        "first_frame_mismatch_index": frame_mismatch,
        "first_inference_step_mismatch_index": step_mismatch,
        "first_observation_payload_mismatch_index": payload_mismatch,
        "first_field_mismatch_index": field_mismatches,
        "first_observation_payload_baseline": first_payload_baseline,
        "first_observation_payload_candidate": first_payload_candidate,
        "status": "passed" if passed else "failed",
        "reason": (
            "same query schedule and observation payloads, including image bytes and robot state"
            if passed
            else "observation parity is not established; do not attribute action differences to PMG"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_observations(
        args.baseline_trace.resolve(), args.candidate_trace.resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
