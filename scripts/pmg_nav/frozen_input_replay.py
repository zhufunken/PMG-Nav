#!/usr/bin/env python3
"""Build an offline M3 shadow trace from one frozen M0 action trace.

This tool deliberately does not import Isaac Sim, TIC-VLA, Torch, or PMG.  It
copies the serialized policy inputs from an existing M0 trace into an M3
diagnostic-only trace and preserves the recorded policy actions.  The result
can prove that a replay artifact carries the same inputs and action checksum;
it cannot prove what an M3 model would have produced in a new rollout.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


REQUIRED_POLICY_PAYLOAD_FIELDS = (
    "instruction",
    "image_refs",
    "robot_state",
    "robot_pose",
    "inference_step",
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    """Read JSONL strictly enough to avoid validating a partial trace."""
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"trace row at {path}:{line_number} is not an object")
        rows.append(item)
    if not rows:
        raise ValueError(f"trace is empty: {path}")
    return rows


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _action_checksum(row: dict[str, Any]) -> str | None:
    """Return the checksum convention used by the generated behavior wrapper."""
    try:
        v_cmd = float(row["v_cmd"])
        w_cmd = float(row["w_cmd"])
    except (KeyError, TypeError, ValueError):
        return None
    return _sha256([v_cmd, w_cmd])


def _mode(row: dict[str, Any]) -> str | None:
    value = row.get("mode")
    return str(value).upper() if value is not None else None


def _resolve_image_ref(image_root: Path | None, ref: Any) -> Path | None:
    if image_root is None or not isinstance(ref, str) or not ref:
        return None
    candidate = Path(ref)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    for path in (image_root / ref, image_root / "images" / ref):
        if path.is_file():
            return path
    return None


def _image_evidence(
    inputs: list[dict[str, Any]], image_root: Path | None
) -> dict[str, Any]:
    refs: list[str] = []
    resolved = 0
    hash_checked = 0
    hash_matches = 0
    hash_mismatches: list[str] = []
    for row in inputs:
        payload = row.get("policy_input_payload")
        if not isinstance(payload, dict):
            continue
        row_refs = payload.get("image_refs")
        row_hashes = payload.get("image_sha256")
        if not isinstance(row_refs, list):
            continue
        refs.extend(str(ref) for ref in row_refs)
        for index, ref in enumerate(row_refs):
            path = _resolve_image_ref(image_root, ref)
            if path is None:
                continue
            resolved += 1
            if not isinstance(row_hashes, list) or index >= len(row_hashes):
                continue
            expected = row_hashes[index]
            if not isinstance(expected, str) or not expected:
                continue
            hash_checked += 1
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest == expected:
                hash_matches += 1
            else:
                hash_mismatches.append(str(ref))
    return {
        "image_ref_count": len(refs),
        "unique_image_ref_count": len(set(refs)),
        "image_root": str(image_root) if image_root else None,
        "resolved_image_ref_count": resolved,
        "content_hash_checked_count": hash_checked,
        "content_hash_match_count": hash_matches,
        "content_hash_mismatch_refs": sorted(set(hash_mismatches)),
        "content_hash_evidence_available": hash_checked > 0,
    }


def _shadow_row(row: dict[str, Any]) -> dict[str, Any]:
    shadow = copy.deepcopy(row)
    source_mode = _mode(row)
    shadow["shadow_replay"] = True
    shadow["shadow_source_mode"] = source_mode
    shadow["mode"] = "M3"
    shadow["pmg_diagnostic_only"] = True
    shadow["pmg_action_authority"] = "bypassed"
    kind = row.get("kind")
    if kind == "input":
        # Keep the serialized policy input untouched.  These flags describe the
        # M3 control branch and are not inserted into TIC-VLA's input payload.
        shadow["pmg_action_enabled"] = False
        shadow["pmg_vector_consumed"] = False
        shadow["pmg_control_policy"] = "policy_only"
        shadow["pmg_payload_present"] = False
    elif kind == "output":
        # The action is a frozen identity replay of the recorded M0 policy
        # action.  Clear PMG authority metadata so readers cannot mistake this
        # artifact for a live M3 intervention.
        shadow["control_policy"] = "policy_only"
        shadow["pmg_override"] = False
        shadow["recovery"] = False
        shadow["pmg_control_source"] = "policy"
        shadow["pmg_gate_scope"] = "policy"
        shadow["pmg_route_enabled"] = False
        shadow["pmg_route_valid"] = False
    return shadow


def build_m3_shadow(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create an M3 shadow while preserving row order and frozen payloads."""
    return [_shadow_row(row) for row in rows]


def _first_mismatch(left: list[Any], right: list[Any]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def validate_shadow(
    source_rows: list[dict[str, Any]],
    shadow_rows: list[dict[str, Any]],
    *,
    source_trace: Path | None = None,
    shadow_trace: Path | None = None,
    image_root: Path | None = None,
) -> dict[str, Any]:
    source_inputs = [row for row in source_rows if row.get("kind") == "input"]
    shadow_inputs = [row for row in shadow_rows if row.get("kind") == "input"]
    source_outputs = [row for row in source_rows if row.get("kind") == "output"]
    shadow_outputs = [row for row in shadow_rows if row.get("kind") == "output"]
    source_inferences = [row for row in source_rows if row.get("kind") == "inference"]
    shadow_inferences = [row for row in shadow_rows if row.get("kind") == "inference"]

    source_payloads = [row.get("policy_input_payload") for row in source_inputs]
    shadow_payloads = [row.get("policy_input_payload") for row in shadow_inputs]
    payload_complete = bool(
        source_inputs
        and all(
            isinstance(payload, dict)
            and all(field in payload for field in REQUIRED_POLICY_PAYLOAD_FIELDS)
            for payload in source_payloads
        )
    )
    payload_mismatch = _first_mismatch(source_payloads, shadow_payloads)
    source_policy_checksums = [
        row.get("policy_input_checksum") for row in source_inputs
    ]
    shadow_policy_checksums = [
        row.get("policy_input_checksum") for row in shadow_inputs
    ]
    policy_checksum_mismatch = _first_mismatch(
        source_policy_checksums, shadow_policy_checksums
    )
    policy_checksum_available = bool(
        source_inputs
        and all(value is not None for value in source_policy_checksums)
    )

    source_commands = [
        (row.get("v_cmd"), row.get("w_cmd")) for row in source_outputs
    ]
    shadow_commands = [
        (row.get("v_cmd"), row.get("w_cmd")) for row in shadow_outputs
    ]
    command_mismatch = _first_mismatch(source_commands, shadow_commands)
    source_checksums = [row.get("checksum") for row in source_outputs]
    shadow_checksums = [row.get("checksum") for row in shadow_outputs]
    checksum_mismatch = _first_mismatch(source_checksums, shadow_checksums)
    recomputed = [_action_checksum(row) for row in source_outputs]
    recorded_checksum_match = [
        expected is not None and expected == recorded
        for expected, recorded in zip(recomputed, source_checksums)
    ]
    action_checksum_schema_available = bool(
        source_outputs
        and all(value is not None for value in recomputed)
        and all(value is not None for value in source_checksums)
    )
    source_modes = sorted({mode for mode in (_mode(row) for row in source_rows) if mode})
    source_is_m0_like = not source_modes or source_modes == ["M0"]
    override_count = sum(
        bool(row.get("pmg_override", row.get("recovery", False)))
        for row in source_outputs
    )
    image_evidence = _image_evidence(source_inputs, image_root)

    structural_passed = bool(
        source_rows
        and len(source_rows) == len(shadow_rows)
        and len(source_inputs) == len(shadow_inputs)
        and len(source_outputs) == len(shadow_outputs)
        and len(source_inferences) == len(shadow_inferences)
        and source_is_m0_like
        and payload_complete
        and source_outputs
        and action_checksum_schema_available
        and all(recorded_checksum_match)
        and payload_mismatch is None
        and policy_checksum_mismatch is None
        and command_mismatch is None
        and checksum_mismatch is None
        and all(row.get("mode") == "M3" for row in shadow_rows if row.get("kind") in {"input", "inference", "output"})
    )
    payload_claim = bool(payload_complete and payload_mismatch is None)
    action_claim = bool(
        action_checksum_schema_available
        and all(recorded_checksum_match)
        and command_mismatch is None
        and checksum_mismatch is None
    )
    status = "passed" if structural_passed else "blocked_or_failed"
    return {
        "schema_version": 1,
        "kind": "stage2_frozen_input_m3_shadow",
        "source_trace": str(source_trace) if source_trace else None,
        "shadow_trace": str(shadow_trace) if shadow_trace else None,
        "status": status,
        "source": {
            "row_count": len(source_rows),
            "input_count": len(source_inputs),
            "output_count": len(source_outputs),
            "inference_count": len(source_inferences),
            "mode_values": source_modes,
            "source_is_m0_like": source_is_m0_like,
            "source_output_override_count": override_count,
        },
        "shadow": {
            "row_count": len(shadow_rows),
            "input_count": len(shadow_inputs),
            "output_count": len(shadow_outputs),
            "inference_count": len(shadow_inferences),
            "target_mode": "M3",
            "action_authority": "bypassed",
        },
        "execution": {
            "isaac_started": False,
            "model_loaded": False,
            "identity_action_replay": True,
            "same_python_process_for_copy_and_validation": True,
        },
        "same_input": {
            "row_count_equal": len(source_rows) == len(shadow_rows),
            "policy_input_payload_complete": payload_complete,
            "policy_input_payload_exact_copy": payload_claim,
            "policy_input_payload_first_mismatch_index": payload_mismatch,
            "policy_input_checksum_available": policy_checksum_available,
            "policy_input_checksum_exact_copy": bool(
                policy_checksum_available and policy_checksum_mismatch is None
            ),
            "policy_input_checksum_first_mismatch_index": policy_checksum_mismatch,
        },
        "action_checksum": {
            "recorded_checksum_schema_available": action_checksum_schema_available,
            "recorded_checksum_match_count": sum(recorded_checksum_match),
            "recorded_checksum_count": len(recorded_checksum_match),
            "recorded_checksum_matches_source": action_claim,
            "shadow_command_exact_copy": command_mismatch is None,
            "shadow_checksum_exact_copy": checksum_mismatch is None,
            "first_command_mismatch_index": command_mismatch,
            "first_checksum_mismatch_index": checksum_mismatch,
        },
        "image_evidence": image_evidence,
        "claims": {
            "validated": [
                "the generated M3 shadow preserves the serialized policy payload and recorded action rows"
                if structural_passed
                else "the shadow artifact was generated, but structural validation did not pass",
                "the action checksum convention is internally consistent"
                if action_claim
                else "the source action checksum could not be fully validated",
            ],
            "not_proven": [
                "an M3 model action was not generated",
                "PMG did not influence a live action decision",
                "closed-loop navigation, SR, SPL, collision, or latency improvement",
                "cross-process observation parity beyond the serialized payload",
            ],
        },
        "reason": (
            "offline identity replay validated; this is not an end-to-end M3 evaluation"
            if structural_passed
            else "missing or inconsistent frozen trace fields; do not interpret as M3 performance"
        ),
    }


def replay(
    source_trace: Path,
    shadow_trace: Path,
    report_path: Path,
    *,
    image_root: Path | None = None,
) -> dict[str, Any]:
    source_rows = _rows(source_trace)
    shadow_rows = build_m3_shadow(source_rows)
    _write_rows(shadow_trace, shadow_rows)
    report = validate_shadow(
        source_rows,
        shadow_rows,
        source_trace=source_trace,
        shadow_trace=shadow_trace,
        image_root=image_root,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--shadow-trace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="optional directory containing canonical image refs; bytes are only hashed, never loaded into a model",
    )
    args = parser.parse_args()
    try:
        report = replay(
            args.source_trace.resolve(),
            args.shadow_trace.resolve(),
            args.report.resolve(),
            image_root=args.image_root.resolve() if args.image_root else None,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
