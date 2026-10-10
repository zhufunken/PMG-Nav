#!/usr/bin/env python3
"""Check timing/action parity between two Isaac/TIC-VLA JSONL traces.

The checker is intentionally dependency-free.  It does not decide whether a
method is better; it only answers whether a purported M0/M3 replay has the
same query schedule, policy-input checksums, and applied commands.  A failed
parity check is a diagnostic gate and must not be converted into a performance
claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _rows(path: Path, kind: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("kind") == kind:
            values.append(item)
    return values


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_mismatch(left: list[Any], right: list[Any]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _load_result_provenance(path: Path | None) -> dict[str, Any] | None:
    """Load the compact parity fields from an episode result JSON.

    Result files have existed in two shapes during Stage 2 (the provenance
    object may occur at the top level and/or inside the first episode).  Keep
    this helper tolerant of both layouts while never treating a missing file
    as proof of parity.
    """
    if path is None:
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    merged: dict[str, Any] = {}
    for provenance_key in ("behavior_provenance", "behavior_execution"):
        top = document.get(provenance_key)
        if isinstance(top, dict):
            merged.update(top)
    episodes = document.get("episodes")
    if isinstance(episodes, list) and episodes and isinstance(episodes[0], dict):
        for provenance_key in ("behavior_provenance", "behavior_execution"):
            nested = episodes[0].get(provenance_key)
            if isinstance(nested, dict):
                # Top-level fields are generally the more complete run-level
                # record; nested fields fill gaps without overwriting them.
                for key, value in nested.items():
                    merged.setdefault(key, value)
        # A few early manual-driver artifacts put timeout metadata directly
        # on the episode object.  Preserve it as a provenance fallback while
        # keeping the strict gate explicit about what was actually available.
        for key in (
            "parity_fixed_horizon",
            "deterministic_torch_cuda",
            "effective_timeout_frames",
            "configured_timeout_frames",
        ):
            if key in episodes[0]:
                merged.setdefault(key, episodes[0][key])
    return merged or None


def _provenance_comparison(
    baseline_result: Path | None,
    candidate_result: Path | None,
) -> dict[str, Any]:
    """Compare run-level parity provenance when result JSONs are supplied."""
    baseline = _load_result_provenance(baseline_result)
    candidate = _load_result_provenance(candidate_result)
    fields = (
        "parity_fixed_horizon",
        "deterministic_torch_cuda",
        "effective_timeout_frames",
        "configured_timeout_frames",
    )
    comparisons: dict[str, Any] = {}
    mismatches: list[str] = []
    for field in fields:
        left = baseline.get(field) if baseline is not None else None
        right = candidate.get(field) if candidate is not None else None
        available = left is not None and right is not None
        equal = bool(available and left == right)
        comparisons[field] = {
            "available": available,
            "baseline": left,
            "candidate": right,
            "equal": equal,
        }
        # Only fields present in both files can establish equality.  Missing
        # provenance is reported as unavailable and is handled by the strict
        # gate below rather than silently accepted.
        if not available or not equal:
            mismatches.append(field)
    return {
        "baseline_result": str(baseline_result) if baseline_result else None,
        "candidate_result": str(candidate_result) if candidate_result else None,
        "available": baseline is not None and candidate is not None,
        "fields": comparisons,
        "mismatch_fields": mismatches,
    }


def check_traces(
    baseline_trace: Path,
    candidate_trace: Path,
    *,
    command_tolerance: float = 1e-7,
    require_greedy: bool = True,
    require_output_checksum: bool = True,
    baseline_result: Path | None = None,
    candidate_result: Path | None = None,
) -> dict[str, Any]:
    baseline_inputs = _rows(baseline_trace, "input")
    candidate_inputs = _rows(candidate_trace, "input")
    baseline_outputs = _rows(baseline_trace, "output")
    candidate_outputs = _rows(candidate_trace, "output")
    baseline_inferences = _rows(baseline_trace, "inference")
    candidate_inferences = _rows(candidate_trace, "inference")

    baseline_policy = [item.get("policy_input_checksum") for item in baseline_inputs]
    candidate_policy = [item.get("policy_input_checksum") for item in candidate_inputs]
    baseline_frames = [item.get("simulation_frame") for item in baseline_inputs]
    candidate_frames = [item.get("simulation_frame") for item in candidate_inputs]
    baseline_steps = [item.get("inference_step") for item in baseline_inputs]
    candidate_steps = [item.get("inference_step") for item in candidate_inputs]
    baseline_policy_payloads = [
        item.get("policy_input_payload")
        for item in baseline_inputs
        if "policy_input_payload" in item
    ]
    candidate_policy_payloads = [
        item.get("policy_input_payload")
        for item in candidate_inputs
        if "policy_input_payload" in item
    ]
    # Optional content hashes are emitted only when a run opts into
    # PMG_TRACE_CONTENT_HASH.  Keep them diagnostic-only so historical traces
    # remain readable, while exposing the first image/robot-state mismatch in
    # new paired runs.
    baseline_content_rows = [
        item for item in baseline_inputs if item.get("trace_content_hash_enabled")
    ]
    candidate_content_rows = [
        item for item in candidate_inputs if item.get("trace_content_hash_enabled")
    ]
    content_hash_evidence_available = bool(
        baseline_content_rows
        and candidate_content_rows
        and len(baseline_content_rows) == len(baseline_inputs)
        and len(candidate_content_rows) == len(candidate_inputs)
    )
    baseline_image_hashes = [
        item.get("image_content_sha256") for item in baseline_content_rows
    ]
    candidate_image_hashes = [
        item.get("image_content_sha256") for item in candidate_content_rows
    ]
    baseline_robot_hashes = [item.get("robot_state_checksum") for item in baseline_content_rows]
    candidate_robot_hashes = [item.get("robot_state_checksum") for item in candidate_content_rows]
    image_content_mismatch = _first_mismatch(baseline_image_hashes, candidate_image_hashes)
    robot_state_mismatch = _first_mismatch(baseline_robot_hashes, candidate_robot_hashes)
    baseline_output_frames = [item.get("simulation_frame") for item in baseline_outputs]
    candidate_output_frames = [item.get("simulation_frame") for item in candidate_outputs]
    baseline_output_checksums = [item.get("checksum") for item in baseline_outputs]
    candidate_output_checksums = [item.get("checksum") for item in candidate_outputs]

    policy_mismatch = _first_mismatch(baseline_policy, candidate_policy)
    frame_mismatch = _first_mismatch(baseline_frames, candidate_frames)
    step_mismatch = _first_mismatch(baseline_steps, candidate_steps)
    output_frame_mismatch = _first_mismatch(
        baseline_output_frames, candidate_output_frames
    )
    output_checksum_mismatch = _first_mismatch(
        baseline_output_checksums, candidate_output_checksums
    )
    policy_payload_available = bool(
        baseline_inputs
        and candidate_inputs
        and len(baseline_policy_payloads) == len(baseline_inputs)
        and len(candidate_policy_payloads) == len(candidate_inputs)
    )
    policy_payload_mismatch = _first_mismatch(
        baseline_policy_payloads, candidate_policy_payloads
    )
    # If only one side emits the payload, identify the first aligned query as
    # the mismatch rather than silently treating a missing preimage as equal.
    if policy_payload_mismatch is None and (
        len(baseline_policy_payloads) != len(candidate_policy_payloads)
    ):
        policy_payload_mismatch = min(
            len(baseline_policy_payloads), len(candidate_policy_payloads)
        )
    policy_payload_presence_mismatch = bool(
        bool(baseline_policy_payloads) != bool(candidate_policy_payloads)
    )
    policy_payload_gate_passed = bool(
        not policy_payload_presence_mismatch
        and (not policy_payload_available or policy_payload_mismatch is None)
    )
    first_policy_payload_baseline = (
        baseline_policy_payloads[policy_payload_mismatch]
        if policy_payload_mismatch is not None
        and policy_payload_mismatch < len(baseline_policy_payloads)
        else None
    )
    first_policy_payload_candidate = (
        candidate_policy_payloads[policy_payload_mismatch]
        if policy_payload_mismatch is not None
        and policy_payload_mismatch < len(candidate_policy_payloads)
        else None
    )

    checksum_available = bool(
        baseline_outputs
        and candidate_outputs
        and all(value is not None for value in baseline_output_checksums)
        and all(value is not None for value in candidate_output_checksums)
    )
    if require_output_checksum and not checksum_available:
        # Use the first missing position as an actionable diagnostic.  A
        # missing checksum must never be interpreted as an equal command.
        if output_checksum_mismatch is None:
            for index, (left, right) in enumerate(
                zip(baseline_output_checksums, candidate_output_checksums)
            ):
                if left is None or right is None:
                    output_checksum_mismatch = index
                    break
            if output_checksum_mismatch is None:
                output_checksum_mismatch = min(
                    len(baseline_output_checksums), len(candidate_output_checksums)
                )

    command_diffs: list[float] = []
    command_mismatch: int | None = None
    for index, (left, right) in enumerate(zip(baseline_outputs, candidate_outputs)):
        left_v, right_v = _finite(left.get("v_cmd")), _finite(right.get("v_cmd"))
        left_w, right_w = _finite(left.get("w_cmd")), _finite(right.get("w_cmd"))
        if None in (left_v, right_v, left_w, right_w):
            command_mismatch = index
            break
        delta = max(abs(left_v - right_v), abs(left_w - right_w))
        command_diffs.append(delta)
        if delta > command_tolerance and command_mismatch is None:
            command_mismatch = index
    if command_mismatch is None and len(baseline_outputs) != len(candidate_outputs):
        command_mismatch = min(len(baseline_outputs), len(candidate_outputs))

    route_enabled_candidate = sum(
        bool(item.get("pmg_route_enabled")) for item in candidate_outputs
    )
    override_candidate = sum(
        bool(item.get("pmg_override", item.get("recovery", False)))
        for item in candidate_outputs
    )
    diagnostic_flags = [
        bool(item.get("parity_greedy_sampling"))
        for item in candidate_inputs
        if "parity_greedy_sampling" in item
    ]
    baseline_diagnostic_flags = [
        bool(item.get("parity_greedy_sampling"))
        for item in baseline_inputs
        if "parity_greedy_sampling" in item
    ]
    greedy_evidence_available = bool(
        baseline_inputs
        and candidate_inputs
        and len(baseline_diagnostic_flags) == len(baseline_inputs)
        and len(diagnostic_flags) == len(candidate_inputs)
    )
    greedy_parity_passed = bool(
        greedy_evidence_available
        and all(baseline_diagnostic_flags)
        and all(diagnostic_flags)
    )
    provenance = _provenance_comparison(baseline_result, candidate_result)
    provenance_passed = not (baseline_result or candidate_result) or bool(
        provenance["available"] and not provenance["mismatch_fields"]
    )

    parity_passed = bool(
        baseline_inputs
        and len(baseline_inputs) == len(candidate_inputs)
        and len(baseline_outputs) == len(candidate_outputs)
        and len(baseline_inferences) == len(candidate_inferences)
        and policy_mismatch is None
        and frame_mismatch is None
        and step_mismatch is None
        and output_frame_mismatch is None
        and command_mismatch is None
        and (not require_output_checksum or output_checksum_mismatch is None)
        and policy_payload_gate_passed
        and (not require_greedy or greedy_parity_passed)
        and provenance_passed
    )
    return {
        "schema_version": 1,
        "kind": "stage2_replay_parity_check",
        "baseline_trace": str(baseline_trace),
        "candidate_trace": str(candidate_trace),
        "baseline": {
            "input_count": len(baseline_inputs),
            "output_count": len(baseline_outputs),
            "inference_count": len(baseline_inferences),
            "policy_input_payload_count": len(baseline_policy_payloads),
            "first_frame": baseline_frames[0] if baseline_frames else None,
            "last_frame": baseline_frames[-1] if baseline_frames else None,
        },
        "candidate": {
            "input_count": len(candidate_inputs),
            "output_count": len(candidate_outputs),
            "inference_count": len(candidate_inferences),
            "policy_input_payload_count": len(candidate_policy_payloads),
            "first_frame": candidate_frames[0] if candidate_frames else None,
            "last_frame": candidate_frames[-1] if candidate_frames else None,
            "route_enabled_output_count": route_enabled_candidate,
            "override_output_count": override_candidate,
            "parity_greedy_sampling": bool(diagnostic_flags)
            and all(diagnostic_flags),
            "parity_greedy_sampling_count": sum(diagnostic_flags),
            "parity_greedy_sampling_evidence_count": len(diagnostic_flags),
        },
        "comparison": {
            "input_count_equal": len(baseline_inputs) == len(candidate_inputs),
            "output_count_equal": len(baseline_outputs) == len(candidate_outputs),
            "inference_count_equal": len(baseline_inferences) == len(candidate_inferences),
            "simulation_frame_equal": frame_mismatch is None,
            "inference_step_equal": step_mismatch is None,
            "output_simulation_frame_equal": output_frame_mismatch is None,
            "policy_input_checksum_equal": policy_mismatch is None,
            "action_command_equal": command_mismatch is None,
            "action_output_checksum_equal": (
                output_checksum_mismatch is None
                if checksum_available
                else False
            ),
            "action_output_checksum_available": checksum_available,
            "policy_input_payload_available": policy_payload_available,
            "policy_input_payload_equal": (
                policy_payload_available and policy_payload_mismatch is None
            ),
            "policy_input_payload_presence_mismatch": policy_payload_presence_mismatch,
            "policy_input_payload_gate_passed": policy_payload_gate_passed,
            "content_hash_evidence_available": content_hash_evidence_available,
            "image_content_hash_equal": (
                content_hash_evidence_available
                and image_content_mismatch is None
            ),
            "robot_state_hash_equal": (
                content_hash_evidence_available
                and robot_state_mismatch is None
            ),
            "first_image_content_hash_mismatch_index": image_content_mismatch,
            "first_robot_state_hash_mismatch_index": robot_state_mismatch,
            "parity_greedy_sampling_required": require_greedy,
            "parity_greedy_sampling_evidence_available": greedy_evidence_available,
            "parity_greedy_sampling_equal": greedy_parity_passed,
            "first_policy_mismatch_index": policy_mismatch,
            "first_frame_mismatch_index": frame_mismatch,
            "first_inference_step_mismatch_index": step_mismatch,
            "first_output_frame_mismatch_index": output_frame_mismatch,
            "first_action_output_checksum_mismatch_index": output_checksum_mismatch,
            "first_policy_input_payload_mismatch_index": policy_payload_mismatch,
            "first_policy_input_payload_baseline": first_policy_payload_baseline,
            "first_policy_input_payload_candidate": first_policy_payload_candidate,
            "first_command_mismatch_index": command_mismatch,
            "max_command_abs_delta": max(command_diffs) if command_diffs else None,
            "command_tolerance": command_tolerance,
        },
        "provenance": provenance,
        "status": "passed" if parity_passed else "failed",
        "reason": (
            "all aligned policy queries, greedy-generation evidence, and applied commands match"
            if parity_passed
            else "trace/action/provenance parity is not established; do not compare performance"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command-tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--allow-stochastic",
        action="store_true",
        help="legacy escape hatch: do not require parity_greedy_sampling=true",
    )
    parser.add_argument(
        "--allow-missing-output-checksum",
        action="store_true",
        help="legacy escape hatch: report but do not gate on output checksums",
    )
    parser.add_argument(
        "--baseline-result",
        type=Path,
        default=None,
        help="optional M0 result JSON for run-level parity provenance",
    )
    parser.add_argument(
        "--candidate-result",
        type=Path,
        default=None,
        help="optional M3 result JSON for run-level parity provenance",
    )
    args = parser.parse_args()
    result = check_traces(
        args.baseline_trace.resolve(),
        args.candidate_trace.resolve(),
        command_tolerance=max(0.0, float(args.command_tolerance)),
        require_greedy=not args.allow_stochastic,
        require_output_checksum=not args.allow_missing_output_checksum,
        baseline_result=args.baseline_result.resolve() if args.baseline_result else None,
        candidate_result=args.candidate_result.resolve() if args.candidate_result else None,
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
