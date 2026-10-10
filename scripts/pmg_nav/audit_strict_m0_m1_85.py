#!/usr/bin/env python3
"""Audit completed strict M0/M1 queue artifacts without running Isaac Sim.

The queue stores one ``benchmark_metrics.json`` below each run directory and
archives action JSONL, telemetry, PMG, and GPU-monitor artifacts beside it.
This checker reads those files, validates the pair contract, and optionally
writes a new audit directory containing a manifest, derived PMG summaries, and
a GPU peak CSV.  Existing source artifacts are never modified or replaced.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


METHODS = ("M0", "M1")
DEFAULT_EPISODES = tuple(range(1, 86))
_METHOD_RE = re.compile(r"(?:^|[/_\\-])m([01])(?:$|[/_\\-])", re.IGNORECASE)
_EPISODE_RE = re.compile(
    r"(?:^|[/_\\-])(?:episode|ep|e)[_\\-]?0*(\\d+)(?:$|[/_\\-])",
    re.IGNORECASE,
)
_NESTED_RESULT_KEYS = ("metrics", "result", "episode_metrics", "benchmark")
_METRIC_KEYS = frozenset(
    {
        "episode",
        "episode_id",
        "episode_key",
        "episode_seed",
        "success",
        "spl",
        "navigation_error",
        "navigation_error_m",
        "artifact_status",
        "error",
    }
)
_TRACE_OUTPUT_FIELDS = (
    "policy_v_cmd",
    "policy_w_cmd",
    "v_cmd",
    "w_cmd",
    "pmg_override",
    "pmg_override_fraction",
    "pmg_control_source",
    "pmg_gate_scope",
    "pmg_gate_reason",
    "pmg_route_valid",
    "pmg_waypoint_index",
    "pmg_route_alignment_hold",
    "pmg_route_stall_policy_fallback",
    "pmg_semantic_candidate",
    "pmg_control_candidate",
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _count_matches(value: Any, observed: int) -> bool:
    """Compare a declared count without allowing malformed metadata to crash the audit."""

    number = _finite(value)
    return number is not None and number.is_integer() and int(number) == observed


def _method(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip().upper()
        if text in METHODS:
            return text
        match = _METHOD_RE.search(value)
        if match:
            return f"M{match.group(1)}"
    return None


def _episode(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        number = int(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        match = re.fullmatch(r"(?:episode|ep|e)[_\-]?0*(\d+)", text)
        if not match and text.isdigit():
            match = re.fullmatch(r"0*(\d+)", text)
        if not match:
            match = _EPISODE_RE.search(text)
        if not match:
            return None
        number = int(match.group(1))
    else:
        return None
    return f"episode_{number}" if 1 <= number <= 999 else None


def _flatten(value: Mapping[str, Any]) -> dict[str, Any]:
    flattened = dict(value)
    for key in _NESTED_RESULT_KEYS:
        nested = value.get(key)
        if isinstance(nested, Mapping):
            for nested_key, nested_value in _flatten(nested).items():
                flattened.setdefault(nested_key, nested_value)
    return flattened


def _record_episode(value: Mapping[str, Any]) -> str | None:
    for key in ("episode", "episode_id", "episode_key", "episode_name"):
        result = _episode(value.get(key))
        if result is not None:
            return result
    for key in ("run_id", "run", "name"):
        result = _episode(value.get(key))
        if result is not None:
            return result
    return None


def _record_method(value: Mapping[str, Any]) -> str | None:
    for key in ("method", "run_method", "policy_method", "mode"):
        result = _method(value.get(key))
        if result is not None:
            return result
    for key in ("run_id", "run", "name"):
        result = _method(value.get(key))
        if result is not None:
            return result
    return None


def _walk_records(
    value: Any,
    *,
    method: str | None = None,
    episode: str | None = None,
    section: tuple[str, ...] = (),
) -> Iterable[tuple[str | None, str | None, dict[str, Any], tuple[str, ...]]]:
    if isinstance(value, Mapping):
        local_method = _record_method(value) or method
        local_episode = _record_episode(value) or episode
        flattened = _flatten(value)
        if local_episode and _METRIC_KEYS.intersection(flattened):
            yield local_method, local_episode, flattened, section
            return
        for key, child in value.items():
            if key in {
                "action_trace",
                "behavior_execution",
                "behavior_provenance",
                "control_gate",
                "frames",
                "observations",
                "pmg",
                "pmg_adapter",
                "telemetry",
                "trace",
            }:
                continue
            yield from _walk_records(
                child,
                method=_method(key) or local_method,
                episode=_episode(key) or local_episode,
                section=section + (str(key),),
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_records(
                child,
                method=method,
                episode=episode,
                section=section + (str(index),),
            )


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _paths(root: Path, name: str, run_id: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    if run_id:
        candidates.extend(
            [
                root / "outputs/dynanav_closed_loop" / run_id / name,
                root / "outputs" / run_id / name,
                root / run_id / name,
            ]
        )
    candidates.extend(root.rglob(name) if root.is_dir() else [])
    unique: list[Path] = []
    for path in candidates:
        if path.is_file() and (run_id is None or run_id in str(path)) and path not in unique:
            unique.append(path)
    return unique


def _resolve_artifact(
    root: Path,
    result_path: Path,
    run_id: str | None,
    requested: Any,
    names: Sequence[str],
) -> Path | None:
    candidates: list[Path] = []
    if isinstance(requested, str) and requested:
        requested_path = Path(requested).expanduser()
        candidates.extend(
            [requested_path, root / requested_path, result_path.parent / requested_path.name]
        )
    for name in names:
        candidates.append(result_path.parent / name)
        if run_id:
            candidates.append(root / "outputs/dynanav_closed_loop" / run_id / name)
        candidates.extend(_paths(root, name, run_id))
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _result_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("benchmark_metrics.json"))


def _load_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload, error = _read_json(path)
    if error or not isinstance(payload, dict):
        raise ValueError(f"cannot read manifest {path}: {error or 'not an object'}")
    return payload


def _expected_selection(
    manifest: Mapping[str, Any],
    explicit_episodes: Iterable[int] | None,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    methods = tuple(
        method
        for method in (_method(item) for item in manifest.get("methods", METHODS))
        if method is not None
    ) or METHODS
    if explicit_episodes is not None:
        episodes = tuple(sorted(set(int(item) for item in explicit_episodes)))
    else:
        keys = manifest.get("official_episode_keys") or manifest.get("episode_keys")
        if isinstance(keys, list) and keys:
            parsed = [_episode(item) for item in keys]
            episodes = tuple(sorted(int(item.rsplit("_", 1)[1]) for item in parsed if item))
        else:
            selection = manifest.get("selection")
            raw = selection.get("episode_indices") if isinstance(selection, Mapping) else None
            episodes = tuple(sorted(set(int(item) for item in raw))) if isinstance(raw, list) and raw else DEFAULT_EPISODES
    return methods, episodes


def _find_episode_candidates(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path in _result_files(root):
        payload, error = _read_json(path)
        if error:
            failures.append({"source": str(path), "reason": "unreadable_benchmark_metrics", "detail": error})
            continue
        path_method = _method(str(path))
        run_id = path.parent.name
        found = list(_walk_records(payload, method=path_method))
        if not found:
            failures.append({"source": str(path), "reason": "no_episode_record"})
        for method, episode, record, section in found:
            candidates.append(
                {
                    "method": method,
                    "episode": episode,
                    "record": record,
                    "payload": payload,
                    "section": list(section),
                    "result_path": str(path),
                    "run_id": str(record.get("run_id") or run_id),
                }
            )
    return candidates, failures


def _number_vector(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    values = [_finite(item) for item in value]
    return [float(item) for item in values] if all(item is not None for item in values) else None


def _vectors_equal(left: Any, right: Any, tolerance: float) -> bool:
    first = _number_vector(left)
    second = _number_vector(right)
    return first is not None and second is not None and all(
        abs(a - b) <= tolerance for a, b in zip(first, second)
    )


def _metric_value(record: Mapping[str, Any], payload: Mapping[str, Any], names: Sequence[str]) -> Any:
    for container in (record, payload):
        for name in names:
            if name in container and container[name] is not None:
                return container[name]
    average = payload.get("average_metrics")
    if isinstance(average, Mapping):
        for name in names:
            if name in average and average[name] is not None:
                return average[name]
    return None


def _protocol_caveats(record: Mapping[str, Any], payload: Mapping[str, Any]) -> list[str]:
    caveats: list[str] = []
    runtime = record.get("compatibility_runner") or payload.get("compatibility_runner")
    if runtime and "compatibility" in str(runtime).lower():
        caveats.append("Isaac Sim compatibility runtime")
    runtime_people = _finite(record.get("runtime_num_people"))
    official_people = _finite(record.get("official_num_people"))
    if runtime_people == 0.0 and (official_people is None or official_people > 0.0):
        caveats.append("runtime_num_people=0; official dynamic people are metadata only")
    if any(record.get(key) is not True and payload.get(key) is not True for key in ("strict_parity_contract", "parity_fixed_horizon", "deterministic_torch_cuda")):
        caveats.append("strict parity provenance is incomplete")
    pmg = record.get("pmg")
    if isinstance(pmg, Mapping) and str(pmg.get("perception_source", "")).lower() in {"oracle", "isaac_usd_oracle_rgbd"}:
        caveats.append("oracle/USD perception condition")
    for key in ("protocol_caveats", "caveats"):
        value = record.get(key, payload.get(key))
        if isinstance(value, str):
            caveats.append(value)
        elif isinstance(value, list):
            caveats.extend(str(item) for item in value if item)
    return list(dict.fromkeys(caveats))


def audit_trace(path: Path | None, expected_summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested_path": str(path) if path else None,
        "path": str(path) if path else None,
        "exists": bool(path and path.is_file()),
        "status": "missing",
        "line_count": 0,
        "valid_json_lines": 0,
        "blank_lines": 0,
        "malformed_lines": [],
        "kind_counts": {},
        "output_count": 0,
        "input_count": 0,
        "inference_count": 0,
        "required_output_missing": [],
        "override_count": 0,
        "override_fraction_derived": None,
        "summary_matches": {},
        "sha256": None,
        "failures": [],
    }
    if path is None or not path.is_file():
        result["failures"].append("action_trace_missing")
        return result
    result["sha256"] = _sha256(path)
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        result["failures"].append(f"action_trace_unreadable:{exc}")
        return result
    result["line_count"] = len(lines)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value}")

    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            result["blank_lines"] += 1
            continue
        try:
            value = json.loads(raw_line, parse_constant=reject_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            result["malformed_lines"].append({"line": line_number, "detail": str(exc)})
            continue
        if not isinstance(value, dict):
            result["malformed_lines"].append({"line": line_number, "detail": "not an object"})
            continue
        rows.append(value)
        kind = str(value.get("kind", "unknown"))
        result["kind_counts"][kind] = result["kind_counts"].get(kind, 0) + 1
    outputs = [row for row in rows if row.get("kind") == "output"]
    result["valid_json_lines"] = len(rows)
    result["output_count"] = len(outputs)
    result["input_count"] = sum(row.get("kind") == "input" for row in rows)
    result["inference_count"] = sum(row.get("kind") == "inference" for row in rows)
    result["override_count"] = sum(bool(row.get("pmg_override", row.get("recovery", False))) for row in outputs)
    result["override_fraction_derived"] = result["override_count"] / max(1, len(outputs)) if outputs else None
    missing: list[dict[str, Any]] = []
    for index, row in enumerate(outputs):
        row_missing = [field for field in _TRACE_OUTPUT_FIELDS if field not in row]
        if row_missing:
            missing.append({"output_index": index, "fields": row_missing})
    result["required_output_missing"] = missing
    expected = expected_summary if isinstance(expected_summary, Mapping) else {}
    checks = {
        "entries": (expected.get("entries"), result["line_count"]),
        "action_output_count": (expected.get("action_output_count"), result["output_count"]),
        "action_input_count": (expected.get("action_input_count"), result["input_count"]),
        "inference_event_count": (expected.get("inference_event_count"), result["inference_count"]),
        "pmg_override_count": (expected.get("pmg_override_count"), result["override_count"]),
    }
    for key, (declared, observed) in checks.items():
        result["summary_matches"][key] = {
            "declared": declared,
            "observed": observed,
            "available": declared is not None,
            "equal": declared is None or _count_matches(declared, observed),
        }
        if declared is not None and not _count_matches(declared, observed):
            result["failures"].append(f"trace_summary_mismatch:{key}")
    declared_fraction = _finite(expected.get("pmg_override_fraction"))
    if declared_fraction is not None and result["override_fraction_derived"] is not None:
        equal = abs(declared_fraction - float(result["override_fraction_derived"])) <= 1e-6
        result["summary_matches"]["pmg_override_fraction"] = {
            "declared": declared_fraction,
            "observed": result["override_fraction_derived"],
            "available": True,
            "equal": equal,
        }
        if not equal:
            result["failures"].append("trace_summary_mismatch:pmg_override_fraction")
    if result["blank_lines"] or result["malformed_lines"]:
        result["failures"].append("trace_contains_invalid_lines")
    if not outputs:
        result["failures"].append("trace_contains_no_output_records")
    if missing:
        result["failures"].append("trace_output_schema_incomplete")
    result["status"] = "pass" if not result["failures"] else "fail"
    return result


def _trace_path(root: Path, result_path: Path, run_id: str, record: Mapping[str, Any]) -> Path | None:
    action_trace = record.get("action_trace")
    requested = action_trace.get("path") if isinstance(action_trace, Mapping) else action_trace
    return _resolve_artifact(root, result_path, run_id, requested, ("action_trace.jsonl",))


def _gpu_summary(root: Path, result_path: Path, run_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
    requested = record.get("gpu_memory_csv") or record.get("gpu_memory_path")
    path = _resolve_artifact(root, result_path, run_id, requested, (f"{run_id}_gpu_memory.csv", "gpu_memory.csv"))
    result: dict[str, Any] = {
        "path": str(path) if path else None,
        "exists": bool(path),
        "rows": 0,
        "invalid_rows": 0,
        "peak_mib": None,
        "benchmark_peak_mib": _finite(record.get("peak_gpu_memory_mib")),
        "failures": [],
    }
    if path is None:
        result["failures"].append("gpu_memory_csv_missing")
        return result
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            values: list[float] = []
            for row in reader:
                result["rows"] += 1
                value = _finite(row.get("memory_used_mib") or row.get("memory_mib") or row.get("peak_gpu_memory_mib"))
                if value is None:
                    result["invalid_rows"] += 1
                else:
                    values.append(value)
    except (OSError, csv.Error) as exc:
        result["failures"].append(f"gpu_memory_csv_unreadable:{exc}")
        return result
    if values:
        result["peak_mib"] = max(values)
    else:
        result["failures"].append("gpu_memory_csv_has_no_numeric_samples")
    if result["invalid_rows"]:
        result["failures"].append("gpu_memory_csv_invalid_rows")
    return result


def _artifacts_summary(root: Path, result_path: Path, run_id: str) -> dict[str, Any]:
    path = _resolve_artifact(root, result_path, run_id, None, ("telemetry/artifacts.json", "artifacts.json"))
    result: dict[str, Any] = {"path": str(path) if path else None, "exists": bool(path), "payload": None, "failures": []}
    if path is None:
        result["failures"].append("telemetry_artifacts_missing")
        return result
    payload, error = _read_json(path)
    if error or not isinstance(payload, Mapping):
        result["failures"].append(f"telemetry_artifacts_unreadable:{error or 'not an object'}")
        return result
    result["payload"] = {
        key: payload.get(key)
        for key in ("status", "rgb_frames", "trajectory_points", "nonzero_command_points", "rgb_width", "rgb_height")
        if key in payload
    }
    if payload.get("status") not in (None, "passed", "valid"):
        result["failures"].append("telemetry_artifacts_status_not_passed")
    for key in ("rgb_frames", "trajectory_points"):
        value = _finite(payload.get(key, 0))
        if value is None or value <= 0.0 or not value.is_integer():
            result["failures"].append(f"telemetry_{key}_missing")
    return result


def _graph_summary(
    root: Path,
    result_path: Path,
    run_id: str,
    episode: str,
    record: Mapping[str, Any],
    method: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pmg = record.get("pmg") if isinstance(record.get("pmg"), Mapping) else {}
    pmg_path_value = pmg.get("graph_path") if isinstance(pmg, Mapping) else None
    graph_path = _resolve_artifact(root, result_path, run_id, pmg_path_value, (f"{episode}.graph.json",))
    graph_payload: Mapping[str, Any] | None = None
    failures: list[str] = []
    if graph_path:
        loaded, error = _read_json(graph_path)
        if error or not isinstance(loaded, Mapping):
            failures.append(f"graph_unreadable:{error or 'not an object'}")
        else:
            graph_payload = loaded
    elif method == "M1":
        failures.append("pmg_graph_source_missing")
    objects = graph_payload.get("objects") if isinstance(graph_payload, Mapping) else None
    relations = graph_payload.get("relations") if isinstance(graph_payload, Mapping) else None
    actual_nodes = len(objects) if isinstance(objects, (Mapping, list)) else None
    actual_edges = len(relations) if isinstance(relations, (Mapping, list)) else None
    summary_nodes = _finite(pmg.get("graph_nodes"))
    summary_edges = _finite(pmg.get("graph_edges"))
    if method == "M1":
        if summary_nodes is None:
            failures.append("pmg_graph_nodes_missing")
        if summary_edges is None:
            failures.append("pmg_graph_edges_missing")
        if actual_nodes is not None and summary_nodes is not None and int(summary_nodes) != actual_nodes:
            failures.append("pmg_graph_nodes_mismatch")
        if actual_edges is not None and summary_edges is not None and int(summary_edges) != actual_edges:
            failures.append("pmg_graph_edges_mismatch")
    graph_mode = pmg.get("graph_mode")
    max_nodes = _finite(pmg.get("max_graph_nodes"))
    max_edges = _finite(pmg.get("max_edges_per_anchor"))
    if graph_mode == "compact" and max_nodes is not None and summary_nodes is not None and summary_nodes > max_nodes:
        failures.append("compact_graph_node_cap_exceeded")
    relation_counts: dict[str, int] = {}
    if isinstance(relations, Mapping):
        for key in relations:
            source = str(key).split("|", 1)[0]
            relation_counts[source] = relation_counts.get(source, 0) + 1
    elif isinstance(relations, list):
        for relation in relations:
            if isinstance(relation, Mapping):
                source = relation.get("source") or relation.get("from") or relation.get("subject")
                if source is not None:
                    relation_counts[str(source)] = relation_counts.get(str(source), 0) + 1
    if graph_mode == "compact" and max_edges is not None and any(count > max_edges for count in relation_counts.values()):
        failures.append("compact_graph_edges_per_anchor_cap_exceeded")

    goal_pose = _finite(pmg.get("goal_pose_error_m"))
    control_position = _number_vector(pmg.get("control_candidate_position"))
    for field in ("goal_pose_error_m", "semantic_goal_pose_error_m", "control_candidate_position", "control_candidate_source", "pinned_anchor_retained"):
        if field not in pmg:
            failures.append(f"pmg_{field}_missing")
    if method == "M1" and goal_pose is None:
        failures.append("pmg_goal_pose_error_missing")
    if method == "M1" and pmg.get("control_candidate_position") is not None and control_position is None:
        failures.append("pmg_control_candidate_position_invalid")
    summary = {
        "source_graph_path": str(graph_path) if graph_path else None,
        "source_graph_sha256": _sha256(graph_path) if graph_path else None,
        "graph_mode": graph_mode,
        "summary_graph_nodes": int(summary_nodes) if summary_nodes is not None else None,
        "summary_graph_edges": int(summary_edges) if summary_edges is not None else None,
        "actual_graph_nodes": actual_nodes,
        "actual_graph_edges": actual_edges,
        "max_graph_nodes": int(max_nodes) if max_nodes is not None else None,
        "max_edges_per_anchor": int(max_edges) if max_edges is not None else None,
        "goal_pose_error_m": goal_pose,
        "semantic_goal_pose_error_m": _finite(pmg.get("semantic_goal_pose_error_m")),
        "control_candidate_position": control_position,
        "control_candidate_source": pmg.get("control_candidate_source"),
        "pinned_anchor_retained": pmg.get("pinned_anchor_retained"),
        "relation_counts_by_source": relation_counts,
        "failures": list(dict.fromkeys(failures)),
        "status": "pass" if not failures else "fail",
    }
    derived = {
        "schema": "strict_m0_m1_85_derived_pmg_graph_v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "episode": episode,
        "run_id": run_id,
        "audit_status": summary["status"],
        "source_graph": summary,
    }
    if graph_payload is not None:
        derived["graph_version"] = graph_payload.get("version")
        derived["graph_association_radius"] = graph_payload.get("association_radius")
    return summary, derived


def _override_summary(record: Mapping[str, Any], trace: Mapping[str, Any], method: str) -> dict[str, Any]:
    control = record.get("control_gate") if isinstance(record.get("control_gate"), Mapping) else {}
    count = _finite(trace.get("override_count")) if trace.get("status") == "pass" else _finite(control.get("pmg_override_count"))
    actions = _finite(trace.get("output_count")) if trace.get("status") == "pass" else _finite(control.get("action_output_count"))
    fraction = _finite(trace.get("override_fraction_derived")) if trace.get("status") == "pass" else _finite(control.get("pmg_override_fraction"))
    if fraction is None and count is not None and actions and actions > 0:
        fraction = count / actions
    failures: list[str] = []
    if count is None:
        failures.append("override_count_missing")
    if actions is None or actions <= 0:
        failures.append("action_output_count_missing")
    if fraction is None or not 0.0 <= fraction <= 1.0:
        failures.append("override_fraction_invalid")
    if method == "M1" and count == 0:
        failures.append("m1_no_override_observed")
    return {
        "count": int(count) if count is not None else None,
        "action_output_count": int(actions) if actions is not None else None,
        "fraction": fraction,
        "control_gate_count": _finite(control.get("pmg_override_count")),
        "control_gate_fraction": _finite(control.get("pmg_override_fraction")),
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }


def _audit_record(
    root: Path,
    artifact: Mapping[str, Any],
    *,
    expected_seed_base: int = 666,
) -> tuple[dict[str, Any], dict[str, Any]]:
    method = artifact.get("method")
    episode = artifact.get("episode")
    record = artifact.get("record") if isinstance(artifact.get("record"), Mapping) else {}
    payload = artifact.get("payload") if isinstance(artifact.get("payload"), Mapping) else {}
    result_path = Path(str(artifact["result_path"]))
    run_id = str(artifact.get("run_id") or result_path.parent.name)
    failures: list[str] = []
    seed = _finite(record.get("episode_seed"))
    episode_number = int(str(episode).rsplit("_", 1)[1]) if isinstance(episode, str) and "_" in episode else None
    expected_seed = expected_seed_base + episode_number if episode_number is not None else None
    if seed is None:
        failures.append("seed_missing")
    elif expected_seed is not None and int(seed) != expected_seed:
        failures.append("seed_mismatch_expected_policy")
    instruction = _metric_value(record, payload, ("instruction",))
    start = _metric_value(record, payload, ("start",))
    goal = _metric_value(record, payload, ("goal",))
    if not isinstance(instruction, str) or not instruction:
        failures.append("instruction_missing")
    if _number_vector(start) is None:
        failures.append("start_invalid")
    if _number_vector(goal) is None:
        failures.append("goal_invalid")
    success = record.get("success")
    spl = _finite(_metric_value(record, payload, ("spl", "mean_spl")))
    navigation_error = _finite(_metric_value(record, payload, ("navigation_error", "navigation_error_m", "nav_error")))
    if not isinstance(success, bool):
        failures.append("success_missing_or_not_bool")
    if spl is None or not 0.0 <= spl <= 1.0:
        failures.append("spl_invalid")
    if navigation_error is None or navigation_error < 0.0:
        failures.append("navigation_error_invalid")
    artifact_status = record.get("artifact_status", payload.get("artifact_status"))
    if artifact_status is not None and artifact_status != "valid":
        failures.append("artifact_status_not_valid")

    trace_path = _trace_path(root, result_path, run_id, record)
    trace_expected = record.get("action_trace") if isinstance(record.get("action_trace"), Mapping) else {}
    trace = audit_trace(trace_path, trace_expected)
    failures.extend(f"action_trace:{item}" for item in trace["failures"])
    artifacts = _artifacts_summary(root, result_path, run_id)
    failures.extend(f"telemetry:{item}" for item in artifacts["failures"])
    gpu = _gpu_summary(root, result_path, run_id, record)
    graph, derived_graph = _graph_summary(root, result_path, run_id, str(episode), record, str(method))
    if method == "M1":
        failures.extend(f"pmg:{item}" for item in graph["failures"])
    override = _override_summary(record, trace, str(method))
    if method == "M1":
        failures.extend(f"override:{item}" for item in override["failures"] if item != "m1_no_override_observed")

    row = {
        "method": method,
        "episode": episode,
        "run_id": run_id,
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
        "status": "pass" if not failures else "fail",
        "failures": list(dict.fromkeys(failures)),
        "metrics": {
            "episode_seed": int(seed) if seed is not None else None,
            "expected_episode_seed": expected_seed,
            "instruction": instruction,
            "start": _number_vector(start),
            "goal": _number_vector(goal),
            "success": success if isinstance(success, bool) else None,
            "spl": spl,
            "navigation_error_m": navigation_error,
            "official_num_people": record.get("official_num_people"),
            "runtime_num_people": record.get("runtime_num_people"),
            "peak_gpu_memory_mib": _finite(record.get("peak_gpu_memory_mib")),
        },
        "action_trace": trace,
        "telemetry_artifacts": artifacts,
        "override": override,
        "pmg_graph": graph,
        "gpu_memory": gpu,
        "protocol_caveats": _protocol_caveats(record, payload),
    }
    return row, derived_graph


def _pair_parity(m0: Mapping[str, Any] | None, m1: Mapping[str, Any] | None, tolerance: float) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "fields": {}, "mismatches": []}
    if not m0 or not m1 or not m0.get("result_path") or not m1.get("result_path"):
        result["mismatches"].append("pair_missing")
        return result
    result["available"] = True
    left = m0.get("metrics", {})
    right = m1.get("metrics", {})
    for field in ("episode_seed", "instruction"):
        equal = left.get(field) == right.get(field) and left.get(field) is not None
        result["fields"][field] = {"m0": left.get(field), "m1": right.get(field), "equal": equal}
        if not equal:
            result["mismatches"].append(field)
    for field in ("start", "goal"):
        equal = _vectors_equal(left.get(field), right.get(field), tolerance)
        result["fields"][field] = {"m0": left.get(field), "m1": right.get(field), "equal": equal, "tolerance": tolerance}
        if not equal:
            result["mismatches"].append(field)
    for field in ("official_num_people", "runtime_num_people"):
        if left.get(field) is not None or right.get(field) is not None:
            equal = left.get(field) == right.get(field)
            result["fields"][field] = {"m0": left.get(field), "m1": right.get(field), "equal": equal}
            if not equal:
                result["mismatches"].append(field)
    result["all_equal"] = not result["mismatches"]
    return result


def audit_queue(
    root: Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    expected_episodes: Iterable[int] | None = None,
    methods: Iterable[str] | None = None,
    progress: Mapping[str, Any] | None = None,
    parity_tolerance: float = 1e-6,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = dict(manifest or {})
    selected_methods, selected_episodes = _expected_selection(manifest, expected_episodes)
    if methods is not None:
        selected_methods = tuple(method for method in (_method(item) for item in methods) if method)
    candidates, discovery_failures = _find_episode_candidates(root)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    discovery = list(discovery_failures)
    for candidate in candidates:
        method = candidate.get("method")
        episode = candidate.get("episode")
        if method not in selected_methods or episode not in {f"episode_{number}" for number in selected_episodes}:
            discovery.append({"source": candidate.get("result_path"), "reason": "out_of_scope_record", "method": method, "episode": episode})
            continue
        grouped.setdefault((str(method), str(episode)), []).append(candidate)

    rows: list[dict[str, Any]] = []
    derived_graphs: list[dict[str, Any]] = []
    for method in selected_methods:
        for number in selected_episodes:
            episode = f"episode_{number}"
            attempts = grouped.get((method, episode), [])
            if not attempts:
                rows.append({
                    "method": method,
                    "episode": episode,
                    "run_id": None,
                    "result_path": None,
                    "status": "missing",
                    "failures": ["benchmark_metrics_missing"],
                    "metrics": {},
                    "protocol_caveats": [],
                    "attempts": [],
                })
                continue
            if len(attempts) > 1:
                discovery.append({"method": method, "episode": episode, "reason": "duplicate_benchmark_attempts", "sources": [item.get("result_path") for item in attempts]})
            # Prefer the sole attempt.  Multiple attempts remain listed and
            # make the row fail closed instead of silently selecting a retry.
            row, derived = _audit_record(root, attempts[0])
            if len(attempts) > 1:
                row["status"] = "fail"
                row["failures"] = list(dict.fromkeys(row["failures"] + ["duplicate_benchmark_attempts"]))
                row["attempts"] = [item.get("result_path") for item in attempts]
            rows.append(row)
            derived_graphs.append(derived)

    by_key = {(row.get("method"), row.get("episode")): row for row in rows}
    pairs: list[dict[str, Any]] = []
    for number in selected_episodes:
        episode = f"episode_{number}"
        m0 = by_key.get(("M0", episode))
        m1 = by_key.get(("M1", episode))
        parity = _pair_parity(m0, m1, parity_tolerance)
        pairs.append({
            "episode": episode,
            "m0_status": m0.get("status") if m0 else "missing",
            "m1_status": m1.get("status") if m1 else "missing",
            "parity": parity,
            "m1_override": m1.get("override") if m1 else None,
        })

    failures = [*discovery]
    for row in rows:
        failures.extend(
            {"method": row.get("method"), "episode": row.get("episode"), "run_id": row.get("run_id"), "reason": reason}
            for reason in row.get("failures", [])
        )
    failures.extend(
        {"episode": pair["episode"], "reason": f"parity:{item}"}
        for pair in pairs
        for item in pair["parity"].get("mismatches", [])
    )
    caveats = list(dict.fromkeys(
        caveat
        for row in rows
        for caveat in row.get("protocol_caveats", [])
    ))
    if any(row.get("status") != "pass" for row in rows):
        caveats.append("incomplete or invalid artifacts remain visible and are not eligible for strict aggregate claims")
    if any(pair["parity"].get("mismatches") for pair in pairs):
        caveats.append("M0/M1 seed/instruction/start/goal parity is not complete for every pair")
    report = {
        "schema": "strict_m0_m1_85_audit_v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "expected_methods": list(selected_methods),
        "expected_episodes": list(selected_episodes),
        "expected_rows": len(selected_methods) * len(selected_episodes),
        "rows": rows,
        "pairs": pairs,
        "complete": len(rows) == len(selected_methods) * len(selected_episodes) and all(row.get("status") == "pass" for row in rows) and all(pair["parity"].get("all_equal") for pair in pairs),
        "valid_rows": sum(row.get("status") == "pass" for row in rows),
        "failures": failures,
        "protocol_caveats": list(dict.fromkeys(caveats)),
        "manifest_metadata": {
            key: manifest.get(key)
            for key in ("run_prefix", "paired_m0_run_prefix", "runtime", "strict_parity", "parity_fixed_horizon", "parity_deterministic", "pairing_seed_policy")
            if key in manifest
        },
        "progress": dict(progress or {}),
    }
    return report, derived_graphs, rows


def _write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_outputs(
    report: Mapping[str, Any],
    derived_graphs: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_new(output_dir / "audit_manifest.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    graph_dir = output_dir / "pmg_graphs"
    for graph in derived_graphs:
        filename = f"{graph.get('method', 'unknown')}_{graph.get('episode', 'unknown')}.json"
        _write_new(graph_dir / filename, json.dumps(graph, indent=2, sort_keys=True) + "\n")
    gpu_path = output_dir / "gpu_peak_summary.csv"
    fieldnames = ("method", "episode", "run_id", "status", "benchmark_peak_mib", "csv_peak_mib", "csv_rows", "csv_path", "gpu_failures")
    gpu_path.parent.mkdir(parents=True, exist_ok=True)
    with gpu_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            gpu = row.get("gpu_memory", {}) if isinstance(row.get("gpu_memory"), Mapping) else {}
            writer.writerow({
                "method": row.get("method"),
                "episode": row.get("episode"),
                "run_id": row.get("run_id"),
                "status": row.get("status"),
                "benchmark_peak_mib": gpu.get("benchmark_peak_mib"),
                "csv_peak_mib": gpu.get("peak_mib"),
                "csv_rows": gpu.get("rows", 0),
                "csv_path": gpu.get("path"),
                "gpu_failures": ";".join(str(item) for item in gpu.get("failures", [])),
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="queue root or a benchmark_metrics.json")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--progress", type=Path, default=None)
    parser.add_argument("--episodes", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="new directory for audit_manifest.json and derived summaries")
    parser.add_argument("--parity-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest = _load_manifest(args.manifest.resolve() if args.manifest else None)
    progress: Mapping[str, Any] = {}
    if args.progress:
        progress_payload, error = _read_json(args.progress.resolve())
        if error or not isinstance(progress_payload, Mapping):
            raise SystemExit(f"cannot read progress JSON: {error or 'not an object'}")
        progress = progress_payload
    report, derived_graphs, rows = audit_queue(
        root,
        manifest=manifest,
        expected_episodes=args.episodes,
        progress=progress,
        parity_tolerance=args.parity_tolerance,
    )
    if args.output_dir:
        write_outputs(report, derived_graphs, rows, args.output_dir.resolve())
    print(json.dumps({
        "complete": report["complete"],
        "valid_rows": report["valid_rows"],
        "expected_rows": report["expected_rows"],
        "failure_count": len(report["failures"]),
        "output_dir": str(args.output_dir.resolve()) if args.output_dir else None,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
