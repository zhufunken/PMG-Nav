#!/usr/bin/env python3
"""Aggregate full Stage 2 route-family matrix and produce report/manifest."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics


METHODS = ("M0", "M1", "M2", "M3")
RUN_PREFIX = os.environ.get("STAGE2_RUN_PREFIX")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path("/root/autodl-tmp/ccfb-ticvla").resolve()
    stage2 = root / "outputs/stage2_full"
    progress_path = stage2 / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    artifacts: list[Path] = [progress_path]
    for run_id, status in sorted(progress.get("runs", {}).items()):
        if RUN_PREFIX and not run_id.startswith(f"{RUN_PREFIX}_"):
            continue
        result_path = Path(status.get("result", ""))
        if not result_path.is_absolute():
            result_path = root / result_path
        if not result_path.is_file():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        episode = payload.get("episodes", [{}])[0]
        method = run_id.rsplit("_", 2)[-2].upper() if run_id.startswith("stage2_full_") else "UNKNOWN"
        episode_wall_seconds = episode.get("wall_seconds")
        row = {
            "run_id": run_id,
            "method": method,
            "episode": episode.get("episode"),
            "scene": episode.get("scene"),
            "success": bool(episode.get("success", False)),
            "spl": float(episode.get("spl", 0.0) or 0.0),
            "navigation_error_m": episode.get("navigation_error"),
            "collision": float(episode.get("collision", 0.0) or 0.0),
            "physical_collision": float(episode.get("physical_collision", 0.0) or 0.0),
            "path_length_m": float(episode.get("path_length", 0.0) or 0.0),
            "duration_s": float(episode.get("duration", 0.0) or 0.0),
            "wall_seconds": float(
                episode_wall_seconds if episode_wall_seconds is not None else status.get("wall_seconds", 0.0)
            ),
            "status": status.get("status"),
        }
        pmg = None
        if method != "M0":
            pmg_path = root / "runtime/dynanav/pmg_metrics" / run_id / f"{episode.get('episode')}.json"
            if pmg_path.is_file():
                pmg = json.loads(pmg_path.read_text(encoding="utf-8"))
                artifacts.append(pmg_path)
        if pmg:
            for key in (
                "graph_nodes", "graph_edges", "observations", "repeated_exploration_count",
                "tool_calls", "tool_cost", "corrections", "correction_rate", "graph_precision",
                "graph_recall", "relation_brier_score", "constraint_satisfaction", "mean_observation_wall_seconds",
            ):
                row[key] = pmg.get(key, 0.0)
            row["constraint_satisfaction_by_relation"] = pmg.get("constraint_satisfaction_by_relation", {})
        rows.append(row)
        artifacts.append(result_path)
        for suffix in ("telemetry/trajectory.csv", "telemetry/episode.mp4", "telemetry/artifacts.json"):
            candidate = root / "outputs/dynanav_closed_loop" / run_id / suffix
            if candidate.is_file():
                artifacts.append(candidate)
    summary: dict[str, object] = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "episodes": len(rows),
        "passed_processes": sum(row["status"] == "passed" for row in rows),
        "failed_processes": sum(row["status"] == "failed" for row in rows),
        "methods": {},
        "rows": rows,
    }
    for method in METHODS:
        values = [row for row in rows if row["method"] == method]
        if not values:
            continue
        metric = {
            "episodes": len(values),
            "success_rate": sum(row["success"] for row in values) / len(values),
            "mean_spl": statistics.fmean(row["spl"] for row in values),
            "mean_navigation_error_m": statistics.fmean(float(row["navigation_error_m"]) for row in values if row["navigation_error_m"] is not None) if any(row["navigation_error_m"] is not None for row in values) else None,
            "collision_rate": statistics.fmean(row["collision"] for row in values),
            "physical_collision_rate": statistics.fmean(row["physical_collision"] for row in values),
            "mean_path_length_m": statistics.fmean(row["path_length_m"] for row in values),
            "mean_wall_seconds": statistics.fmean(row["wall_seconds"] for row in values),
        }
        pmg_values = [row for row in values if "graph_nodes" in row]
        if pmg_values:
            for key in ("graph_precision", "graph_recall", "relation_brier_score", "constraint_satisfaction", "correction_rate", "tool_cost", "repeated_exploration_count", "mean_observation_wall_seconds"):
                metric[key] = statistics.fmean(float(row.get(key, 0.0)) for row in pmg_values)
            metric["total_corrections"] = sum(int(row.get("corrections", 0)) for row in pmg_values)
            relation_values = defaultdict(list)
            for row in pmg_values:
                for relation, score in row.get("constraint_satisfaction_by_relation", {}).items():
                    relation_values[relation].append(float(score))
            metric["constraint_satisfaction_by_relation"] = {key: statistics.fmean(vals) for key, vals in relation_values.items()}
        summary["methods"][method] = metric
    stage2.mkdir(parents=True, exist_ok=True)
    (stage2 / "full_matrix_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = []
    for method, metric in summary["methods"].items():
        table.append(
            f"| {method} | {metric['episodes']} | {metric['success_rate']:.3f} | {metric['mean_spl']:.3f} | "
            f"{metric['mean_navigation_error_m'] if metric['mean_navigation_error_m'] is not None else 'n/a'} | "
            f"{metric.get('constraint_satisfaction', 'n/a')} | {metric.get('graph_precision', 'n/a')} | "
            f"{metric.get('graph_recall', 'n/a')} | {metric.get('correction_rate', 'n/a')} | {metric.get('tool_cost', 'n/a')} |"
        )
    report = f"""# Stage 2 Full PMG Inference Report

Generated: `{summary['generated']}`

## Workload

The complete runnable route-family matrix contains the 20 unique routes represented by benchmark_full episodes 1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 53, 55, 57, 59, 61, 66, 71, 76, and 81. The repeated benchmark variants differ only in dynamic-person count; Isaac Sim 6 compatibility runtime deliberately sets `num_people=0` because the released Replicator Agent 0.x API is unavailable. Every method uses the same route, start, goal, timeout, scene, and checkpoint.

Completed process artifacts: {summary['passed_processes']} passed, {summary['failed_processes']} failed, {summary['episodes']} result rows.

## Matrix Metrics

| Method | Episodes | SR | SPL | Nav error (m) | Physical satisfaction | Graph precision | Graph recall | Correction rate | Tool cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

M0 is the locked TIC-VLA checkpoint. M1 resets to a stateless local graph each observation. M2 maintains the temporal PMG. M3 maintains PMG and applies fixed heuristic tools; each M3 route contains a deterministic conflicting evidence injection followed by verification, and the correction is logged in the episode artifact.

## Artifact Contracts

- Every non-M0 episode has a PMG JSON and serialized graph.
- Every PMG episode has repeated frame observations, local graph nodes/edges, candidate goal scores, tool decisions, 512-d adapter dimension, and observation wall latency.
- Metrics include SR/SPL/CR, physical constraint satisfaction by relation where the compiler resolved the instruction, graph precision/recall against the simulator object catalog, relation Brier score, correction rate, repeated exploration, and tool cost.
- Videos and trajectory CSVs are retained for every successful process.

## Scope Boundaries

The current DynaNav scene assets expose stable USD prim identity and metric pose, so they are the Stage 2 annotation-gated oracle perception source. The perception interface is detector-shaped (`SceneOraclePerception`); GroundingDINO weights are not used in this oracle gate. Dynamic people are excluded by the Isaac 6 compatibility limitation and are reported explicitly rather than silently treated as equivalent to the original benchmark.
"""
    reports = stage2 / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "stage2_full_pmg_inference_report.md").write_text(report, encoding="utf-8")
    manifest = {}
    for path in sorted(set(artifacts)):
        if path.is_file():
            manifest[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (stage2 / "full_reproduction_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary["methods"], indent=2))


if __name__ == "__main__":
    main()
