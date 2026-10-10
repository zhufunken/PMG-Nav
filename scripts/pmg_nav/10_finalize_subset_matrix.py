#!/usr/bin/env python3
"""Aggregate a paired representative subset before any full benchmark run.

The official 85-episode finalizer is intentionally separate from this small
summary.  A subset must have one valid result for every method/episode pair,
and every non-baseline method must have its PMG and serialized graph artifact.
The resulting JSON is the only input accepted by ``09_subset_gate.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from subset_utils import load_manifest, manifest_episode_indices, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/ccfb-ticvla"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--progress",
        type=Path,
        default=None,
        help="runner progress JSON; defaults to outputs/stage2_official_compat/progress.json",
    )
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="run id prefix; defaults to the runner workload manifest, then STAGE2_RUN_PREFIX",
    )
    return parser.parse_args()


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute() and path.is_file():
        return path
    return root / path


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _matches_people(episode: dict, expected: object) -> bool:
    """Keep the official crowd stratum bound to the selected episode."""

    if expected is None:
        return True
    try:
        return int(episode.get("official_num_people", -1)) == int(expected) and int(
            episode.get("runtime_num_people", -1)
        ) == 0
    except (TypeError, ValueError):
        return False


def _artifacts_complete(root: Path, run_id: str) -> bool:
    path = root / "outputs/dynanav_closed_loop" / run_id / "telemetry/artifacts.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        int(payload.get("rgb_frames", 0) or 0) > 0
        and int(payload.get("trajectory_points", 0) or 0) > 0
    )


def _load_result(root: Path, status: dict, run_id: str) -> tuple[dict | None, Path | None]:
    candidate = status.get("result")
    path = _resolve(root, candidate) if candidate else root / "outputs/dynanav_closed_loop" / run_id / "benchmark_metrics.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        episode = payload.get("episodes", [])[0]
    except (OSError, TypeError, IndexError, AttributeError, json.JSONDecodeError):
        return None, path
    return (episode if isinstance(episode, dict) else None), path


def _load_pmg(root: Path, run_id: str, episode_key: str, method: str) -> tuple[dict | None, bool]:
    path = root / "runtime/dynanav/pmg_metrics" / run_id / f"{episode_key}.json"
    graph = path.with_suffix(".graph.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return None, False
    valid = (
        isinstance(payload, dict)
        and payload.get("method") == method
        and int(payload.get("observations", 0) or 0) >= 2
        and int(payload.get("graph_nodes", 0) or 0) >= 1
        and graph.is_file()
    )
    return payload if valid else None, valid


def _method_summary(rows: list[dict], method: str) -> dict:
    values = [row for row in rows if row.get("method") == method and row.get("valid")]
    def mean(key: str) -> float | None:
        numbers = [_finite(row.get(key)) for row in values]
        finite = [value for value in numbers if value is not None]
        return statistics.fmean(finite) if finite else None

    return {
        "episodes": len(values),
        "success_rate": mean("success"),
        "mean_spl": mean("spl"),
        "mean_navigation_error_m": mean("navigation_error_m"),
        "collision_rate": mean("collision"),
        "mean_wall_seconds": mean("wall_seconds"),
    }


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    methods = [str(method) for method in manifest.get("methods", ("M0", "M1"))]
    indices = manifest_episode_indices(manifest)
    manifest_people = {
        item.get("episode"): item.get("official_num_people")
        for item in manifest.get("episodes", [])
        if isinstance(item, dict) and isinstance(item.get("episode"), str)
    }
    progress_path = (args.progress or (root / "outputs/stage2_official_compat/progress.json")).resolve()
    workload_prefix = None
    workload_path = progress_path.parent / "official_workload_manifest.json"
    try:
        workload_payload = json.loads(workload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        workload_payload = {}
    if isinstance(workload_payload, dict) and isinstance(workload_payload.get("run_prefix"), str):
        workload_prefix = workload_payload["run_prefix"]
    prefix = (
        args.run_prefix
        or workload_prefix
        or os.environ.get("STAGE2_RUN_PREFIX")
        or "stage2_subset_core"
    )
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read progress JSON {progress_path}: {exc}") from exc
    runs = progress.get("runs", {}) if isinstance(progress, dict) else {}

    rows: list[dict] = []
    missing: list[str] = []
    for method in methods:
        for index in indices:
            episode_key = f"episode_{index}"
            run_id = f"{prefix}_{method.lower()}_e{index:03d}"
            status = runs.get(run_id, {}) if isinstance(runs, dict) else {}
            status_ok = status.get("status") in {"passed", "cached"}
            episode, result_path = _load_result(root, status, run_id) if status_ok else (None, None)
            valid = bool(
                status_ok
                and episode is not None
                and episode.get("episode") == episode_key
                and not episode.get("error")
                and _matches_people(episode, manifest_people.get(episode_key))
                and isinstance(episode.get("success"), bool)
                and _finite(episode.get("navigation_error")) is not None
                and _finite(episode.get("spl")) is not None
                and 0.0 <= float(episode["spl"]) <= 1.0
                and _artifacts_complete(root, run_id)
            )
            pmg = None
            if method != "M0" and valid:
                pmg, pmg_valid = _load_pmg(root, run_id, episode_key, method)
                valid = valid and pmg_valid
            if not valid:
                missing.append(run_id)
            row = {
                "run_id": run_id,
                "method": method,
                "episode": index,
                "episode_key": episode_key,
                "valid": valid,
                "status": status.get("status", "untracked"),
                "result": str(result_path) if result_path is not None else None,
                "official_num_people": episode.get("official_num_people") if episode else None,
                "runtime_num_people": episode.get("runtime_num_people") if episode else None,
                "success": bool(episode.get("success", False)) if episode else None,
                "spl": _finite(episode.get("spl")) if episode else None,
                "navigation_error_m": _finite(episode.get("navigation_error")) if episode else None,
                "collision": _finite(episode.get("collision", 0.0)) if episode else None,
                "wall_seconds": _finite(episode.get("wall_seconds", status.get("wall_seconds", 0.0))) if episode else None,
            }
            if pmg is not None:
                for key in (
                    "graph_nodes", "graph_edges", "observations", "constraint_satisfaction",
                    "graph_precision", "graph_recall", "correction_rate", "tool_cost",
                ):
                    row[key] = pmg.get(key)
            rows.append(row)

    summary: dict[str, object] = {
        "schema_version": 1,
        "kind": "pmg_nav_subset_matrix_summary",
        "generated": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "progress": str(progress_path),
        "run_prefix": prefix,
        "methods": {method: _method_summary(rows, method) for method in methods},
        "rows": rows,
        "expected_rows": len(methods) * len(indices),
        "completed_rows": sum(bool(row["valid"]) for row in rows),
        "missing_or_invalid": missing,
        "complete": not missing and len(rows) == len(methods) * len(indices),
        "selection": manifest.get("selection", {}),
    }
    output = args.output.resolve()
    write_json(output, summary)
    print(json.dumps({key: summary[key] for key in ("complete", "expected_rows", "completed_rows", "missing_or_invalid", "methods")}, indent=2))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
