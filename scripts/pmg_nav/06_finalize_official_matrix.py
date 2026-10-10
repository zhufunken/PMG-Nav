#!/usr/bin/env python3
"""Aggregate a manifest-selected Stage 2 compatibility matrix.

With no manifest this preserves the historical 85-episode behavior; a
subset/workload manifest narrows the same validation and pairing logic to the
registered core MVP workload.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from subset_utils import load_manifest, load_yaml_mapping, manifest_episode_indices, sha256_file


METHODS = ("M0", "M1", "M2", "M3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/root/autodl-tmp/ccfb-ticvla"),
        help="TIC-VLA/Isaac workspace root (default: /root/autodl-tmp/ccfb-ticvla)",
    )
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        default=None,
        help="deterministic subset manifest used by the runner",
    )
    parser.add_argument(
        "--workload-manifest",
        type=Path,
        default=None,
        help="runner workload manifest; defaults to outputs/stage2_official_compat/official_workload_manifest.json",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help="benchmark_full.yaml; defaults to the path in the workload manifest or the standard root path",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=None,
        help="method set to aggregate; defaults to the workload/subset manifest",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="result directory; defaults to <root>/outputs/stage2_official_compat",
    )
    return parser.parse_args()


def _bootstrap_ci(values: list[float], seed: int = 20260822, samples: int = 2000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(samples)]
    means.sort()
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


def _wilcoxon_signed_rank_pvalue(values: list[float]) -> float | None:
    """Return an exact two-sided signed-rank p-value without SciPy.

    Zero differences are discarded, tied absolute differences receive their
    average rank, and a small integer dynamic program enumerates the signed
    rank-sum distribution.  The largest supported matrix here is 85 pairs,
    for which the state space is only O(n^2).
    """

    differences = [float(value) for value in values if math.isfinite(float(value)) and float(value) != 0.0]
    if not differences:
        return None
    absolute = sorted((abs(value), index) for index, value in enumerate(differences))
    ranks = [0.0] * len(differences)
    cursor = 0
    while cursor < len(absolute):
        end = cursor + 1
        while end < len(absolute) and absolute[end][0] == absolute[cursor][0]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[absolute[position][1]] = average_rank
        cursor = end

    # Multiply ranks by two so tied average ranks remain exact integers.
    rank2 = [int(round(rank * 2.0)) for rank in ranks]
    observed = sum(rank for value, rank in zip(differences, rank2) if value > 0.0)
    total_rank = sum(rank2)
    distribution = [0] * (total_rank + 1)
    distribution[0] = 1
    for rank in rank2:
        for subtotal in range(total_rank, -1, -1):
            count = distribution[subtotal]
            if count:
                distribution[subtotal + rank] += count
    total_assignments = 2 ** len(rank2)
    lower = sum(distribution[: observed + 1])
    upper = sum(distribution[observed:])
    return min(1.0, 2.0 * min(lower, upper) / total_assignments)


def _mean(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(numeric) if numeric else None


def _load_progress(output_root: Path) -> dict:
    merged: dict[str, dict] = {}
    for path in sorted(output_root.glob("progress*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for run_id, status in payload.get("runs", {}).items():
            old = merged.get(run_id)
            # A retry is allowed to replace an earlier failed process, but a
            # completed valid process must not be downgraded by a stale shard.
            if old is None or (status.get("status") == "passed" and old.get("status") != "passed"):
                merged[run_id] = status
    return merged


def _read_json_object(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must contain a JSON object: {path}")
    return payload


def _episode_indices_from_workload(workload: dict) -> list[int] | None:
    keys = workload.get("official_episode_keys")
    if not isinstance(keys, list) or not keys:
        return None
    indices: list[int] = []
    for key in keys:
        if not isinstance(key, str) or not key.startswith("episode_"):
            raise SystemExit(f"invalid workload episode key: {key!r}")
        try:
            index = int(key.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            raise SystemExit(f"invalid workload episode key: {key!r}") from None
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise SystemExit("workload manifest contains duplicate episode ids")
    return indices


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_root = (args.output_root or root / "outputs/stage2_official_compat").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    workload_path = (args.workload_manifest or output_root / "official_workload_manifest.json").resolve()
    workload = _read_json_object(workload_path, "workload manifest") if workload_path.is_file() else {}

    benchmark_value = args.benchmark or workload.get("benchmark_config")
    official_config = (
        Path(benchmark_value).expanduser()
        if isinstance(benchmark_value, (str, Path))
        else root / "src/TIC-VLA/DynaNav/configs/benchmark_full.yaml"
    )
    if not official_config.is_absolute():
        official_config = root / official_config
    official_config = official_config.resolve()
    if not official_config.is_file():
        raise SystemExit(f"benchmark config not found: {official_config}")
    official_source = load_yaml_mapping(official_config)

    subset_path = args.subset_manifest
    if subset_path is None:
        selection = workload.get("selection")
        if isinstance(selection, dict) and isinstance(selection.get("subset_manifest"), str):
            candidate = Path(selection["subset_manifest"])
            if not candidate.is_absolute():
                candidate = root / candidate
            if candidate.is_file():
                subset_path = candidate
    subset_manifest = load_manifest(subset_path.resolve()) if subset_path is not None else None
    if subset_manifest is not None:
        expected_hash = subset_manifest.get("benchmark_sha256")
        if expected_hash and expected_hash != sha256_file(official_config):
            raise SystemExit("subset manifest benchmark_sha256 does not match the benchmark config")

    subset_indices = manifest_episode_indices(subset_manifest) if subset_manifest is not None else None
    workload_indices = _episode_indices_from_workload(workload)
    if subset_indices is not None and workload_indices is not None and subset_indices != workload_indices:
        raise SystemExit("workload manifest episode selection does not match --subset-manifest")
    episode_indices = subset_indices or workload_indices or list(range(1, 86))
    if any(index < 1 or f"episode_{index}" not in official_source for index in episode_indices):
        raise SystemExit("workload/subset manifest selects an episode absent from the benchmark")

    manifest_methods = workload.get("methods")
    if not isinstance(manifest_methods, list) and subset_manifest is not None:
        manifest_methods = subset_manifest.get("methods")
    # A legacy shard writes a workload manifest containing only the method it
    # was asked to run.  That is not a complete matrix declaration, so retain
    # the manifest method set only when it includes the paired M0 baseline;
    # otherwise aggregate the full declared method family and report missing
    # rows rather than silently producing a single-method report.
    default_methods = (
        manifest_methods
        if isinstance(manifest_methods, list) and len(manifest_methods) > 1 and "M0" in manifest_methods
        else METHODS
    )
    methods = list(args.methods or default_methods)
    if not methods or len(set(methods)) != len(methods):
        raise SystemExit("methods must be a non-empty list without duplicates")
    if any(method not in METHODS for method in methods):
        raise SystemExit(f"unsupported method in manifest: {methods!r}")
    if "M0" not in methods:
        raise SystemExit("M0 must be included as the paired baseline")

    run_prefix = os.environ.get("STAGE2_RUN_PREFIX") or workload.get("run_prefix") or "stage2_official"
    m0_run_prefix = os.environ.get("STAGE2_M0_PREFIX", run_prefix)
    runs = _load_progress(output_root)
    rows: list[dict] = []
    missing: list[str] = []
    for method in methods:
        for index in episode_indices:
            episode_key = f"episode_{index}"
            # M0 is a locked baseline and may be reused from a prior complete
            # official run. M1-M3 always use the current experiment prefix.
            method_prefix = m0_run_prefix if method == "M0" else run_prefix
            run_id = f"{method_prefix}_{method.lower()}_e{index:03d}"
            result_path = root / "outputs/dynanav_closed_loop" / run_id / "benchmark_metrics.json"
            status = runs.get(run_id, {})
            if status.get("status") not in {"passed", "cached"}:
                missing.append(f"{run_id}:process_status={status.get('status', 'untracked')}")
                continue
            if not result_path.is_file():
                missing.append(run_id)
                continue
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                episode = payload["episodes"][0]
            except (OSError, KeyError, IndexError, json.JSONDecodeError):
                missing.append(run_id)
                continue
            expected_people = int(official_source[episode_key].get("num_people", 0))
            recorded_seed = episode.get("episode_seed")
            seed_valid = (
                recorded_seed is not None
                and int(recorded_seed) == int(official_source.get("seed", 666)) + index
            )
            # The locked v5 M0 baseline predates seed metadata in its result
            # JSON. Reusing it is explicitly allowed, but the omission remains
            # visible in the row/report and is never applied to M1-M3.
            legacy_m0_without_seed = method == "M0" and m0_run_prefix != run_prefix and recorded_seed is None
            if (
                episode.get("episode") != episode_key
                or episode.get("error")
                or episode.get("artifact_status", "valid") != "valid"
                or int(episode.get("official_num_people", -1)) != expected_people
                or int(episode.get("runtime_num_people", -1)) != 0
                or not (seed_valid or legacy_m0_without_seed)
            ):
                missing.append(run_id)
                continue
            try:
                navigation_error = float(episode["navigation_error"])
                spl = float(episode["spl"])
                duration = float(episode["duration"])
            except (KeyError, TypeError, ValueError):
                missing.append(f"{run_id}:metrics")
                continue
            if (
                not isinstance(episode.get("success"), bool)
                or not all(math.isfinite(value) for value in (navigation_error, spl, duration))
                or navigation_error < 0.0
                or not 0.0 <= spl <= 1.0
                or duration < 0.0
            ):
                missing.append(f"{run_id}:metrics_invalid")
                continue
            artifacts_path = root / "outputs/dynanav_closed_loop" / run_id / "telemetry/artifacts.json"
            try:
                artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                missing.append(f"{run_id}:artifacts")
                continue
            if int(artifacts.get("rgb_frames", 0) or 0) <= 0 or int(artifacts.get("trajectory_points", 0) or 0) <= 0:
                missing.append(f"{run_id}:artifacts_incomplete")
                continue
            row = {
                "run_id": run_id,
                "method": method,
                "episode": index,
                "official_num_people": int(episode.get("official_num_people", -1)),
                "runtime_num_people": int(episode.get("runtime_num_people", -1)),
                "success": bool(episode.get("success", False)),
                "spl": spl,
                "navigation_error_m": navigation_error,
                "collision": float(episode.get("collision", 0.0) or 0.0),
                "physical_collision": float(episode.get("physical_collision", 0.0) or 0.0),
                "duration_s": duration,
                "wall_seconds": float(episode.get("wall_seconds", status.get("wall_seconds", 0.0)) or 0.0),
                "process_status": status.get("status", "untracked"),
                "seed_metadata_missing": bool(legacy_m0_without_seed),
                "diagnostic_contract": episode.get("diagnostic_contract"),
                "action_trace": episode.get("action_trace"),
                "control_gate": episode.get("control_gate"),
                "peak_gpu_memory_mib": episode.get("peak_gpu_memory_mib"),
            }
            control_gate = episode.get("control_gate")
            if isinstance(control_gate, dict):
                def _trace_number(key: str):
                    value = control_gate.get(key)
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        return None
                    return value if math.isfinite(value) else None
                row["control_policy"] = control_gate.get("control_policy")
                row["pmg_override_fraction"] = _trace_number("pmg_override_fraction")
                row["pmg_override_count"] = _trace_number("pmg_override_count")
                row["pmg_terminal_override_count"] = _trace_number("pmg_terminal_override_count")
                row["pmg_global_override_count"] = _trace_number("pmg_global_override_count")
                row["pmg_detour_override_count"] = _trace_number("pmg_detour_override_count")
                row["policy_action_count"] = _trace_number("policy_action_count")
                row["pmg_route_valid_count"] = _trace_number("pmg_route_valid_count")
                row["pmg_route_output_count"] = _trace_number("pmg_route_output_count")
                row["pmg_route_deferred_output_count"] = _trace_number("pmg_route_deferred_output_count")
                row["pmg_route_alignment_hold_count"] = _trace_number("pmg_route_alignment_hold_count")
                row["pmg_route_alignment_motion_violation_count"] = _trace_number("pmg_route_alignment_motion_violation_count")
                row["pmg_route_stall_policy_fallback_count"] = _trace_number("pmg_route_stall_policy_fallback_count")
                row["pmg_route_heading_error_max_abs_rad"] = _trace_number("pmg_route_heading_error_max_abs_rad")
                # Canonical policy-query checksum is the M0/M3 parity field;
                # PMG payload checksum is a separate data-flow audit field.
                row["policy_input_checksum"] = control_gate.get("policy_input_checksum")
                row["policy_input_checksum_count"] = _trace_number("policy_input_checksum_count")
                row["policy_input_checksum_unique_count"] = _trace_number("policy_input_checksum_unique_count")
                row["action_input_checksum"] = control_gate.get("action_input_checksum")
                row["action_output_checksum"] = control_gate.get("action_output_checksum")
                row["action_output_count"] = _trace_number("action_output_count")
                row["parity_greedy_sampling"] = bool(control_gate.get("parity_greedy_sampling", False))
                row["parity_greedy_sampling_count"] = _trace_number("parity_greedy_sampling_count")
                row["pmg_payload_checksum"] = control_gate.get("pmg_payload_checksum")
                row["pmg_payload_checksum_count"] = _trace_number("pmg_payload_checksum_count")
                row["pmg_payload_checksum_unique_count"] = _trace_number("pmg_payload_checksum_unique_count")
                row["simulation_frame_first"] = _trace_number("simulation_frame_first")
                row["simulation_frame_last"] = _trace_number("simulation_frame_last")
                row["inference_event_count"] = _trace_number("inference_event_count")
            else:
                row.update({
                    "control_policy": None,
                    "pmg_override_fraction": None,
                    "pmg_override_count": None,
                    "pmg_terminal_override_count": None,
                    "pmg_global_override_count": None,
                    "pmg_detour_override_count": None,
                    "policy_action_count": None,
                    "pmg_route_valid_count": None,
                    "pmg_route_output_count": None,
                    "pmg_route_deferred_output_count": None,
                    "pmg_route_alignment_hold_count": None,
                    "pmg_route_alignment_motion_violation_count": None,
                    "pmg_route_stall_policy_fallback_count": None,
                    "pmg_route_heading_error_max_abs_rad": None,
                    "policy_input_checksum": None,
                    "policy_input_checksum_count": None,
                    "policy_input_checksum_unique_count": None,
                    "action_input_checksum": None,
                    "action_output_checksum": None,
                    "action_output_count": None,
                    "parity_greedy_sampling": False,
                    "parity_greedy_sampling_count": None,
                    "pmg_payload_checksum": None,
                    "pmg_payload_checksum_count": None,
                    "pmg_payload_checksum_unique_count": None,
                    "simulation_frame_first": None,
                    "simulation_frame_last": None,
                    "inference_event_count": None,
                })
            if method != "M0":
                pmg_path = root / "runtime/dynanav/pmg_metrics" / run_id / f"{episode_key}.json"
                try:
                    pmg = json.loads(pmg_path.read_text(encoding="utf-8"))
                    graph_path = pmg_path.with_suffix(".graph.json")
                    pmg_valid = (
                        pmg.get("method") == method
                        and int(pmg.get("observations", 0) or 0) >= 2
                        and int(pmg.get("graph_nodes", 0) or 0) >= 1
                        and graph_path.is_file()
                    )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    pmg_valid = False
                    pmg = {}
                if not pmg_valid:
                    row["pmg_missing"] = True
                    missing.append(f"{run_id}:pmg")
                    continue
                for key in (
                    "graph_precision", "graph_recall", "relation_brier_score",
                    "constraint_satisfaction", "correction_rate", "tool_cost",
                    "repeated_exploration_count", "mean_observation_wall_seconds",
                    "graph_catalog_recall", "invalid_graph_nodes",
                    "observations", "corrections",
                    "metric_constraint_satisfaction", "relation_constraint_satisfaction",
                    "orientation_satisfaction", "reachability_satisfaction", "metric_error_m",
                    "goal_pose_error_m", "correction_success", "verification_revisit_cost",
                ):
                    value = pmg.get(key)
                    row[key] = (float(value) if value is not None else None)
                row["action_enabled"] = bool(pmg.get("action_enabled", True))
                row["perception_source"] = pmg.get("perception_source")
                row["pmg_vector_consumed"] = bool(pmg.get("pmg_vector_consumed", False))
            rows.append(row)

    expected_rows = len(methods) * len(episode_indices)
    summary: dict = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "official_episode_count": len(episode_indices),
        "benchmark_episode_count": sum(1 for key in official_source if str(key).startswith("episode_")),
        "selected_episode_indices": episode_indices,
        "methods_requested": methods,
        "intended_processes": expected_rows,
        "completed_result_rows": len(rows),
        "missing_or_invalid": sorted(set(missing)),
        "complete": len(rows) == expected_rows and not missing,
        "run_prefix": run_prefix,
        "m0_run_prefix": m0_run_prefix,
        "workload_manifest": str(workload_path) if workload else None,
        "subset_manifest": str(subset_path.resolve()) if subset_path is not None else None,
        "subset_manifest_sha256": sha256_file(subset_path.resolve()) if subset_path is not None else None,
        "selection": workload.get("selection", {}),
        "fairness_contract": {
            "evaluator_goal_available_to_pmg": False,
            "evaluator_goal_used_for_success_only": True,
            "runtime_dynamic_people": 0,
        },
        "runtime_note": "Isaac Sim 6 compatibility: official_num_people retained; runtime_num_people=0 because Replicator Agent 0.x is unavailable",
        "methods": {},
        "by_official_num_people": {},
        "paired_deltas_vs_m0": {},
        "rows": rows,
        "legacy_m0_seed_metadata_missing": sum(
            int(row.get("seed_metadata_missing", False)) for row in rows if row["method"] == "M0"
        ),
    }
    for method in methods:
        values = [row for row in rows if row["method"] == method]
        success = [float(row["success"]) for row in values]
        spl = [row["spl"] for row in values]
        nav = [float(row["navigation_error_m"]) for row in values if row["navigation_error_m"] is not None]
        metric = {
            "episodes": len(values),
            "success_rate": _mean(success),
            "mean_spl": _mean(spl),
            "mean_navigation_error_m": _mean(nav),
            "collision_rate": _mean([row["collision"] for row in values]),
            "physical_collision_rate": _mean([row["physical_collision"] for row in values]),
            "mean_duration_s": _mean([row["duration_s"] for row in values]),
            "mean_wall_seconds": _mean([row["wall_seconds"] for row in values]),
            "peak_gpu_memory_mib": _mean([float(row["peak_gpu_memory_mib"]) for row in values if row.get("peak_gpu_memory_mib") is not None]),
        }
        for key in (
            "graph_precision", "graph_recall", "relation_brier_score",
            "constraint_satisfaction", "correction_rate", "tool_cost",
            "repeated_exploration_count", "mean_observation_wall_seconds",
            "graph_catalog_recall", "invalid_graph_nodes",
            "metric_constraint_satisfaction", "relation_constraint_satisfaction",
            "orientation_satisfaction", "reachability_satisfaction", "metric_error_m",
            "goal_pose_error_m", "verification_revisit_cost",
            "pmg_override_fraction", "pmg_override_count",
            "pmg_terminal_override_count", "pmg_global_override_count",
            "pmg_detour_override_count", "policy_action_count",
            "pmg_route_valid_count", "pmg_route_output_count",
            "pmg_route_deferred_output_count",
            "pmg_route_alignment_hold_count",
            "pmg_route_alignment_motion_violation_count",
            "pmg_route_stall_policy_fallback_count",
            "pmg_route_heading_error_max_abs_rad",
            "policy_input_checksum_count", "policy_input_checksum_unique_count",
            "pmg_payload_checksum_count", "pmg_payload_checksum_unique_count",
        ):
            candidates = [row[key] for row in values if key in row]
            if candidates:
                metric[key] = _mean(candidates)
        summary["methods"][method] = metric

    by_people: dict[str, dict[str, dict]] = defaultdict(dict)
    for people in sorted({row["official_num_people"] for row in rows}):
        for method in methods:
            values = [row for row in rows if row["official_num_people"] == people and row["method"] == method]
            if values:
                by_people[str(people)][method] = {
                    "episodes": len(values),
                    "success_rate": _mean([float(row["success"]) for row in values]),
                    "mean_spl": _mean([row["spl"] for row in values]),
                    "mean_navigation_error_m": _mean([float(row["navigation_error_m"]) for row in values if row["navigation_error_m"] is not None]),
                }
    summary["by_official_num_people"] = dict(by_people)

    indexed = {(row["method"], row["episode"]): row for row in rows}
    # Query-parity audit is intentionally independent of task metrics.  A
    # matching policy-input checksum means M0/M3 presented the same canonical
    # instruction/image query sequence; PMG payload hashes are reported only
    # as an audit and must not be used as a parity criterion.
    parity_checks: dict[str, dict] = {}
    for method in methods:
        if method == "M0":
            continue
        comparisons = []
        for index in episode_indices:
            new_row = indexed.get((method, index))
            base_row = indexed.get(("M0", index))
            if not new_row or not base_row:
                continue
            new_trace = new_row.get("action_trace") if isinstance(new_row.get("action_trace"), dict) else {}
            base_trace = base_row.get("action_trace") if isinstance(base_row.get("action_trace"), dict) else {}
            new_gate = new_row.get("control_gate") if isinstance(new_row.get("control_gate"), dict) else {}
            base_gate = base_row.get("control_gate") if isinstance(base_row.get("control_gate"), dict) else {}
            # Prefer top-level control-gate fields, then action-trace fields for
            # artifacts produced before the gate was promoted to top level.
            def _trace_value(row_trace, row_gate, key):
                return row_gate.get(key) if row_gate.get(key) is not None else row_trace.get(key)
            new_policy = _trace_value(new_trace, new_gate, "policy_input_checksum")
            base_policy = _trace_value(base_trace, base_gate, "policy_input_checksum")
            new_count = _trace_value(new_trace, new_gate, "policy_input_checksum_count")
            base_count = _trace_value(base_trace, base_gate, "policy_input_checksum_count")
            if new_count is None:
                new_count = new_trace.get("action_input_count")
            if base_count is None:
                base_count = base_trace.get("action_input_count")
            new_action_checksum = _trace_value(new_trace, new_gate, "action_output_checksum")
            base_action_checksum = _trace_value(base_trace, base_gate, "action_output_checksum")
            new_action_count = _trace_value(new_trace, new_gate, "action_output_count")
            base_action_count = _trace_value(base_trace, base_gate, "action_output_count")
            if new_action_count is None:
                new_action_count = new_trace.get("action_output_count")
            if base_action_count is None:
                base_action_count = base_trace.get("action_output_count")
            new_greedy = _trace_value(new_trace, new_gate, "parity_greedy_sampling")
            base_greedy = _trace_value(base_trace, base_gate, "parity_greedy_sampling")
            new_frame_first = _trace_value(new_trace, new_gate, "simulation_frame_first")
            base_frame_first = _trace_value(base_trace, base_gate, "simulation_frame_first")
            new_frame_last = _trace_value(new_trace, new_gate, "simulation_frame_last")
            base_frame_last = _trace_value(base_trace, base_gate, "simulation_frame_last")
            action_checksum_available = bool(new_action_checksum and base_action_checksum)
            greedy_evidence_available = new_greedy is not None and base_greedy is not None
            try:
                action_count_equal = bool(
                    new_action_count is not None
                    and base_action_count is not None
                    and int(float(new_action_count)) == int(float(base_action_count))
                )
            except (TypeError, ValueError, OverflowError):
                action_count_equal = False
            comparisons.append({
                "episode": index,
                "available": bool(new_policy and base_policy),
                "policy_input_checksum_equal": bool(new_policy and base_policy and new_policy == base_policy),
                "policy_input_checksum_m1": new_policy,
                "policy_input_checksum_m0": base_policy,
                "policy_input_checksum_count_m1": new_count,
                "policy_input_checksum_count_m0": base_count,
                "action_output_count_m1": new_trace.get("action_output_count"),
                "action_output_count_m0": base_trace.get("action_output_count"),
                "action_output_checksum_m1": new_action_checksum,
                "action_output_checksum_m0": base_action_checksum,
                "action_output_checksum_available": action_checksum_available,
                "action_output_checksum_equal": bool(
                    action_checksum_available and new_action_checksum == base_action_checksum
                ),
                "action_output_count_equal": action_count_equal,
                "parity_greedy_sampling_m1": new_greedy,
                "parity_greedy_sampling_m0": base_greedy,
                "parity_greedy_sampling_evidence_available": greedy_evidence_available,
                "parity_greedy_sampling_equal": bool(
                    greedy_evidence_available and bool(new_greedy) and bool(base_greedy)
                ),
                "simulation_frame_first_m1": new_frame_first,
                "simulation_frame_first_m0": base_frame_first,
                "simulation_frame_last_m1": new_frame_last,
                "simulation_frame_last_m0": base_frame_last,
            })
        available = [item for item in comparisons if item["available"]]
        strict_available = [
            item for item in available
            if item["action_output_checksum_available"]
            and item["parity_greedy_sampling_evidence_available"]
        ]
        legacy_status = "passed" if available and all(
            item["policy_input_checksum_equal"] for item in available
        ) else ("not_available" if not available else "failed")
        strict_status = "passed" if (
            available
            and len(strict_available) == len(available)
            and all(
            item["policy_input_checksum_equal"]
            and item["action_output_checksum_equal"]
            and item["action_output_count_equal"]
            and item["parity_greedy_sampling_equal"]
            for item in strict_available
            )
        ) else ("not_available" if not strict_available else "failed")
        parity_checks[method] = {
            "paired_episodes": len(comparisons),
            "checksum_available_episodes": len(available),
            "checksum_equal_episodes": sum(bool(item["policy_input_checksum_equal"]) for item in available),
            "checksum_equal_fraction": (
                sum(bool(item["policy_input_checksum_equal"]) for item in available) / len(available)
                if available else None
            ),
            # ``legacy_status`` preserves the old policy-only audit for
            # traceability; ``status`` is now strict and cannot pass without
            # action checksum and greedy-generation evidence.
            "legacy_status": legacy_status,
            "strict_available_episodes": len(strict_available),
            "action_output_checksum_equal_episodes": sum(
                bool(item["action_output_checksum_equal"]) for item in strict_available
            ),
            "greedy_evidence_equal_episodes": sum(
                bool(item["parity_greedy_sampling_equal"]) for item in strict_available
            ),
            "status": strict_status,
            "rows": comparisons,
        }
    summary["parity_checks"] = parity_checks
    for method in methods:
        if method == "M0":
            continue
        paired = [(indexed[(method, i)], indexed[("M0", i)]) for i in episode_indices if (method, i) in indexed and ("M0", i) in indexed]
        spl_delta = [new["spl"] - base["spl"] for new, base in paired]
        nav_delta = [float(new["navigation_error_m"]) - float(base["navigation_error_m"]) for new, base in paired if new["navigation_error_m"] is not None and base["navigation_error_m"] is not None]
        sr_delta = [float(new["success"]) - float(base["success"]) for new, base in paired]
        summary["paired_deltas_vs_m0"][method] = {
            "paired_episodes": len(paired),
            "success_rate_delta": _mean(sr_delta),
            "success_rate_delta_bootstrap_95ci": _bootstrap_ci(sr_delta),
            "mean_spl_delta": _mean(spl_delta),
            "mean_spl_delta_bootstrap_95ci": _bootstrap_ci(spl_delta),
            "mean_navigation_error_delta_m": _mean(nav_delta),
            "mean_navigation_error_delta_bootstrap_95ci": _bootstrap_ci(nav_delta),
            "success_rate_wilcoxon_pvalue": _wilcoxon_signed_rank_pvalue(sr_delta),
            "mean_spl_wilcoxon_pvalue": _wilcoxon_signed_rank_pvalue(spl_delta),
            "mean_navigation_error_wilcoxon_pvalue": _wilcoxon_signed_rank_pvalue(nav_delta),
            "improved_success_episodes": sum(delta > 0 for delta in sr_delta),
            "worsened_success_episodes": sum(delta < 0 for delta in sr_delta),
        }

    summary_name = "official_matrix_summary.json" if run_prefix == "stage2_official" else f"{run_prefix}_matrix_summary.json"
    (output_root / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 2 官方标准测试集评测与 Baseline 对比报告",
        "",
        f"生成时间：`{summary['generated']}`",
        "",
        "## 1. 最终状态",
        "",
        f"- 完整性结论：**{'通过' if summary['complete'] else '未通过'}**。",
        f"- 有效结果：**{len(rows)}/{expected_rows}**；缺失或无效：**{len(set(missing))}**。",
        f"- 方法矩阵：{', '.join(methods)}；每种方法 {len(episode_indices)} 个 episode。",
        "",
        "只有 `complete=true` 且所有请求方法的 episode 均齐全时，汇总指标才可用于下一步 gate 或最终报告。",
        "",
        "## 2. 评测口径",
        "",
        f"- 官方配置：`DynaNav/configs/benchmark_full.yaml`，基准共 {summary['benchmark_episode_count']} 个 episode；本次选择 {len(episode_indices)} 个，目标矩阵 {expected_rows} 个进程。",
        f"- M0 使用锁定的 TIC-VLA checkpoint，是本轮 Baseline；M0 前缀为 `{m0_run_prefix}`，当前方法前缀为 `{run_prefix}`。",
        "- 公平性约束：真实 goal 只用于 evaluator 的 success/navigation-error 计算，不参与 PMG perception、候选排序或 recovery。",
        "- 当前运行时：Isaac Sim 6 compatibility runner。每条结果同时保留官方 `official_num_people` 和实际 `runtime_num_people=0`；不能将其解释为动态人群官方闭环结果。",
        f"- 复用的 legacy M0 基线有 {summary['legacy_m0_seed_metadata_missing']} 条 artifact 未记录 `episode_seed`；未改写其指标，除 M0 外的方法仍逐条校验 `666 + episode_index`。",
        "- 为避免 Isaac Sim 6 RTX 显存耗尽，outdoor_small 的部分方法条目在失败重试阶段可能使用兼容低分辨率；episode、checkpoint、evaluator 和 PMG 逻辑保持不变。",
        "- 有效条目必须同时具备可解析 benchmark JSON、有限数值指标、RGB 帧、轨迹点；除 M0 外的方法还必须具备方法匹配的 PMG JSON 和 graph JSON。",
        "- 所有显著性判断都使用同一 episode 上相对 M0 的配对 bootstrap（2,000 次，95% CI）。",
        "",
        "## 3. 总体指标",
        "",
        "| 方法 | 条数 | SR | SPL | 平均导航误差 (m) | 碰撞率 | 图精度 | 图召回 | 约束满足率 | 纠错率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        m = summary["methods"].get(method, {})
        def fmt(key: str, digits: int = 3) -> str:
            value = m.get(key)
            return "n/a" if value is None else f"{value:.{digits}f}"
        lines.append(
            f"| {method} | {int(m.get('episodes', 0))} | {fmt('success_rate')} | {fmt('mean_spl')} | {fmt('mean_navigation_error_m')} | {fmt('collision_rate')} | {fmt('graph_precision')} | {fmt('graph_recall')} | {fmt('constraint_satisfaction')} | {fmt('correction_rate')} |"
        )
    lines.extend([
        "",
        "### 3.1 PMG 控制门统计（实际 action trace）",
        "",
        "override fraction 的分母是全部 action tick；它不把 TIC-VLA 的全局路线控制隐藏在稀疏 PMG 日志之外。",
        "",
        "| 方法 | 控制策略 | 平均 PMG override fraction | PMG terminal override | PMG global override | PMG detour override | TIC-VLA policy action |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for method in methods:
        m = summary["methods"].get(method, {})
        policy_values = [
            row.get("control_policy")
            for row in rows
            if row.get("method") == method and row.get("control_policy") is not None
        ]
        policy_text = policy_values[0] if policy_values else "n/a"
        def gate_fmt(key: str) -> str:
            value = m.get(key)
            return "n/a" if value is None else f"{float(value):.3f}"
        lines.append(
            f"| {method} | {policy_text} | {gate_fmt('pmg_override_fraction')} | "
            f"{gate_fmt('pmg_terminal_override_count')} | {gate_fmt('pmg_global_override_count')} | "
            f"{gate_fmt('pmg_detour_override_count')} | {gate_fmt('policy_action_count')} |"
        )
    lines.extend([
        "",
        "### 3.1a M0/M3 查询 parity 审计",
        "",
        "`policy_input_checksum` 只覆盖 instruction、规范化后的图像引用和查询时序字段；" 
        "它不包含 PMG payload。`pmg_payload_checksum` 仅用于证明旁路数据流，不能当作 parity。"
        "严格状态还要求 action output checksum、action 数量和 greedy-generation 证据；"
        "`legacy_status` 仅保留旧的 policy-only 审计。",
        "",
        "| 方法 | 配对条数 | strict 可用 | policy checksum 相等 | action checksum 相等 | greedy 证据相等 | strict 状态 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for method in methods:
        if method == "M0":
            continue
        parity = summary.get("parity_checks", {}).get(method, {})
        lines.append(
            f"| {method} | {int(parity.get('paired_episodes', 0) or 0)} | "
            f"{int(parity.get('strict_available_episodes', 0) or 0)} | "
            f"{int(parity.get('checksum_equal_episodes', 0) or 0)} / {int(parity.get('checksum_available_episodes', 0) or 0)} | "
            f"{int(parity.get('action_output_checksum_equal_episodes', 0) or 0)} | "
            f"{int(parity.get('greedy_evidence_equal_episodes', 0) or 0)} | "
            f"{parity.get('status', 'not_available')} |"
        )
    lines.extend([
        "",
        "### 3.2 图目录覆盖（M1/M2/M3）",
        "",
        "图精度以完整静态 USD 目录为真值集合，图召回保持当前局部感知窗口口径；目录召回用于衡量全场景覆盖。",
        "",
        "| 方法 | 目录召回 | 无效图节点 |",
        "|---|---:|---:|",
    ])
    for method in methods:
        if method == "M0":
            continue
        m = summary["methods"].get(method, {})
        catalog_recall = m.get("graph_catalog_recall")
        catalog_text = "n/a" if catalog_recall is None else f"{float(catalog_recall):.3f}"
        lines.append(f"| {method} | {catalog_text} | {int(m.get('invalid_graph_nodes', 0.0) or 0)} |")
    lines.extend(["", "## 4. 相对 M0 的配对差值", "", "正的 SR/SPL 差值表示提升，负的导航误差差值表示提升。Wilcoxon 为去除零差值后的精确双侧 signed-rank p-value。", "", "| 方法 | 配对条数 | SR 差值 | SR 95% CI | SR Wilcoxon p | SPL 差值 | SPL 95% CI | SPL Wilcoxon p | 导航误差差值 (m) | 导航误差 95% CI | 导航误差 Wilcoxon p |", "|---|---:|---:|---|---:|---:|---|---:|---:|---|---:|"])
    for method in methods:
        if method == "M0":
            continue
        d = summary["paired_deltas_vs_m0"].get(method, {})
        def ci_text(key: str) -> str:
            ci = d.get(key)
            return "n/a" if not ci or ci[0] is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        def delta_text(key: str) -> str:
            value = d.get(key)
            return "n/a" if value is None else f"{value:.3f}"
        def p_text(key: str) -> str:
            value = d.get(key)
            return "n/a" if value is None else f"{float(value):.6f}"
        lines.append(
            f"| {method} | {d.get('paired_episodes', 0)} | {delta_text('success_rate_delta')} | {ci_text('success_rate_delta_bootstrap_95ci')} | {p_text('success_rate_wilcoxon_pvalue')} | "
            f"{delta_text('mean_spl_delta')} | {ci_text('mean_spl_delta_bootstrap_95ci')} | {p_text('mean_spl_wilcoxon_pvalue')} | "
            f"{delta_text('mean_navigation_error_delta_m')} | {ci_text('mean_navigation_error_delta_bootstrap_95ci')} | {p_text('mean_navigation_error_wilcoxon_pvalue')} |"
        )
    lines.extend(["", "## 5. 是否显著优于 Baseline", ""])
    for method in methods:
        if method == "M0":
            continue
        d = summary["paired_deltas_vs_m0"].get(method, {})
        sr_ci = d.get("success_rate_delta_bootstrap_95ci")
        spl_ci = d.get("mean_spl_delta_bootstrap_95ci")
        nav_ci = d.get("mean_navigation_error_delta_bootstrap_95ci")
        sr_supported = bool(sr_ci and sr_ci[0] is not None and sr_ci[0] > 0)
        spl_supported = bool(spl_ci and spl_ci[0] is not None and spl_ci[0] > 0)
        nav_supported = bool(nav_ci and nav_ci[1] is not None and nav_ci[1] < 0)
        overall = "支持" if summary["complete"] and sr_supported and spl_supported and nav_supported else "未证实"
        lines.append(
            f"- **{method}**：总体判定 **{overall} 优于 M0**；SR={'支持' if sr_supported else '未证实'}，"
            f"SPL={'支持' if spl_supported else '未证实'}，导航误差={'支持' if nav_supported else '未证实'}。"
        )
    lines.extend([
        "",
        "“显著提升”要求 SR 与 SPL 的 95% CI 下界均大于 0，同时导航误差差值的 95% CI 上界小于 0。任何一项不满足，都不能表述为该方法在所有主要端到端指标上显著优于 Baseline。不得用测试集反复调阈值，也不预先承诺一定提升。",
        "",
        "## 6. 本轮实现与完整性门禁",
        "",
        "- 官方 instruction 保持权威，PMG 不注入文本 prompt；PMG 候选只通过结构化 recovery 字段传递。",
        "- 候选使用多帧共识与短滞回，控制端仅允许置信度、硬约束、稳定性和距离门禁全部通过的有界末端 recovery。",
        f"- runner 与 finalizer 使用同一套数值、telemetry、PMG JSON 和 graph JSON 校验，失败或不完整条目不得计入 {expected_rows} 条结果。",
        "- 原始逐 episode 数值、分层人数结果和 bootstrap 区间保存在同目录的 matrix summary JSON。",
        "",
        "## 7. 外部泛化候选（未加入本轮实验）",
        "",
        "候选数据集见仓库中的 `reports/stage2_external_generalization_datasets.md`；首选为 HM3D + VLN-CE 与 TEACh。二者未下载、未加入本轮训练，也未用于官方测试集调参。",
        "",
        "## 8. 已知限制",
        "",
        "- Isaac Sim 6 compatibility runtime 无法实例化 released Replicator Agent 0.x，因此实际动态人数为 0；`human collision=0` 不能作为动态避障证据。",
        f"- 本次 {len(episode_indices)}-episode 矩阵只能检验本轮固定实现；若置信区间跨 0，需要在训练/验证集上改进后另起独立前缀复验，不能挑选或覆盖测试结果。",
    ])
    report_name = "stage2_official_compat_report.md" if run_prefix == "stage2_official" else f"{run_prefix}_report.md"
    (report_dir / report_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["methods"], indent=2))
    if not summary["complete"]:
        raise SystemExit(f"official matrix incomplete: {len(rows)}/{expected_rows} valid rows, {len(set(missing))} missing/invalid")


if __name__ == "__main__":
    main()
