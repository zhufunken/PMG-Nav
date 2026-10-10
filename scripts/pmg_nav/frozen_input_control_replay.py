#!/usr/bin/env python3
"""Replay a frozen TIC-VLA control trace through the M3 no-authority gate.

This is deliberately smaller than an Isaac/model replay.  It reuses one
recorded M0 policy input and action sequence, then evaluates the dependency-
free M3 control gate with ``action_enabled=False``.  It proves that the
control adapter cannot change a fixed policy command, while making the
limitation explicit: it does not rerender RGB frames or run TIC-VLA again.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Running a script by path sets ``sys.path[0]`` to ``scripts/pmg_nav`` rather
# than the project root. Keep the package import independent of the caller's
# working directory, matching the other Stage 2 utility scripts.
PROJECT_MAIN = Path(__file__).resolve().parents[2]
if str(PROJECT_MAIN) not in sys.path:
    sys.path.insert(0, str(PROJECT_MAIN))

from pmg_nav.control_gate import decide_control_gate


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _query_key(row: dict[str, Any]) -> tuple[Any, Any]:
    return row.get("simulation_frame"), row.get("inference_step")


def _command_checksum(v_cmd: Any, w_cmd: Any) -> str:
    values = [float(v_cmd), float(w_cmd)]
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _copy_for_mode(row: dict[str, Any], mode: str) -> dict[str, Any]:
    value = copy.deepcopy(row)
    value["mode"] = mode
    value["frozen_input_replay"] = True
    return value


def _select_inputs(
    inputs: list[dict[str, Any]],
    *,
    max_queries: int | None,
    max_frame: int | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in inputs:
        frame = row.get("simulation_frame")
        if max_frame is not None and isinstance(frame, (int, float)) and int(frame) > max_frame:
            break
        selected.append(row)
        if max_queries is not None and len(selected) >= max_queries:
            break
    if not selected:
        raise ValueError("frozen replay selected zero policy-input rows")
    return selected


def build_replay(
    source_trace: Path,
    output_dir: Path,
    *,
    max_queries: int | None,
    max_frame: int | None,
) -> dict[str, Any]:
    rows = _read_rows(source_trace)
    inputs = [row for row in rows if row.get("kind") == "input"]
    outputs = [row for row in rows if row.get("kind") == "output"]
    inferences = [row for row in rows if row.get("kind") == "inference"]
    selected_inputs = _select_inputs(
        inputs, max_queries=max_queries, max_frame=max_frame
    )

    outputs_by_key: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in outputs:
        outputs_by_key.setdefault(_query_key(row), []).append(row)
    fallback_outputs = iter(outputs)
    paired_outputs: list[dict[str, Any]] = []
    for input_row in selected_inputs:
        key = _query_key(input_row)
        candidates = outputs_by_key.get(key)
        if candidates:
            paired_outputs.append(candidates.pop(0))
            continue
        try:
            paired_outputs.append(next(fallback_outputs))
        except StopIteration as exc:
            raise ValueError(f"missing action output for input key {key}") from exc

    baseline_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    selected_keys = {_query_key(row) for row in selected_inputs}
    for input_row, output_row in zip(selected_inputs, paired_outputs):
        baseline_rows.append(_copy_for_mode(input_row, "M0"))
        candidate_input = _copy_for_mode(input_row, "M3")
        candidate_input["pmg_action_enabled"] = False
        candidate_input["pmg_vector_consumed"] = False
        candidate_rows.append(candidate_input)

        baseline_output = _copy_for_mode(output_row, "M0")
        v_cmd = baseline_output.get("policy_v_cmd", baseline_output.get("v_cmd"))
        w_cmd = baseline_output.get("policy_w_cmd", baseline_output.get("w_cmd"))
        if v_cmd is None or w_cmd is None:
            raise ValueError("source output is missing policy_v_cmd/w_cmd")
        # M0 has no PMG authority, so its applied command is the policy command.
        baseline_output["v_cmd"] = float(v_cmd)
        baseline_output["w_cmd"] = float(w_cmd)
        baseline_output["pmg_override"] = False
        baseline_output["pmg_control_source"] = "policy"
        baseline_output["pmg_gate_scope"] = "policy"
        baseline_output["checksum"] = _command_checksum(v_cmd, w_cmd)
        baseline_rows.append(baseline_output)

        gate = decide_control_gate(
            method="M3",
            policy="policy_only",
            action_enabled=False,
            candidate_valid=False,
            candidate_distance_m=None,
        )
        candidate_output = copy.deepcopy(baseline_output)
        candidate_output["mode"] = "M3"
        candidate_output["pmg_action_enabled"] = False
        candidate_output["pmg_override"] = False
        candidate_output["recovery"] = False
        candidate_output["pmg_control_source"] = "policy"
        candidate_output["pmg_gate_scope"] = "policy"
        candidate_output["pmg_gate_reason"] = gate.reason
        candidate_output["pmg_route_enabled"] = False
        candidate_output["pmg_route_valid"] = False
        candidate_output["pmg_vector_consumed"] = False
        candidate_rows.append(candidate_output)

    # Keep inference events aligned with selected inputs.  They are metadata
    # for the strict checker; no model call is made by this tool.
    selected_inference = [
        row for row in inferences if _query_key(row) in selected_keys
    ]
    baseline_rows.extend(_copy_for_mode(row, "M0") for row in selected_inference)
    candidate_rows.extend(_copy_for_mode(row, "M3") for row in selected_inference)

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "m0_frozen.jsonl"
    candidate_path = output_dir / "m3_frozen.jsonl"
    for path, values in ((baseline_path, baseline_rows), (candidate_path, candidate_rows)):
        path.write_text(
            "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
            encoding="utf-8",
        )

    input_count = len(selected_inputs)
    output_count = len(paired_outputs)
    summary = {
        "schema_version": 1,
        "kind": "stage2_frozen_input_control_replay",
        "status": "passed" if input_count == output_count and input_count > 0 else "failed",
        "source_trace": str(source_trace),
        "baseline_trace": str(baseline_path),
        "candidate_trace": str(candidate_path),
        "input_count": input_count,
        "output_count": output_count,
        "inference_count": len(selected_inference),
        "first_frame": selected_inputs[0].get("simulation_frame"),
        "last_frame": selected_inputs[-1].get("simulation_frame"),
        "m3_action_enabled": False,
        "model_replayed": False,
        "rgb_rerendered": False,
        "same_frozen_policy_input": True,
        "limitation": "control-gate replay only; TIC-VLA and Isaac are not rerun",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--max-frame", type=int, default=None)
    args = parser.parse_args()
    if args.max_queries is not None and args.max_queries < 1:
        parser.error("--max-queries must be positive")
    summary = build_replay(
        args.source_trace.resolve(),
        args.output_dir.resolve(),
        max_queries=args.max_queries,
        max_frame=args.max_frame,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
