#!/usr/bin/env python3
"""Create a non-destructive audit manifest for a strict M0/M1 queue.

The runner already writes benchmark metrics and action traces.  This utility
does not rerun or mutate an experiment; it records the exact source paths,
trace hashes, compact PMG graph summary, GPU peak reported by the benchmark,
and paired parity fields in a new output directory.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


RUN_RE = re.compile(r"_m([01])_e(\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trace_stats(path_value: Any) -> tuple[bool, int, str]:
    if not isinstance(path_value, str):
        return False, 0, ""
    path = Path(path_value)
    if not path.is_file():
        return False, 0, ""
    lines = 0
    with path.open("rb") as handle:
        for _ in handle:
            lines += 1
    return True, lines, sha256_file(path)


def episode_from_file(path: Path) -> tuple[str, int, dict[str, Any]]:
    match = RUN_RE.search(path.parent.name)
    if match is None:
        raise ValueError(f"cannot infer method/episode from {path}")
    method = f"M{match.group(1)}"
    episode = int(match.group(2))
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes or not isinstance(episodes[0], dict):
        raise ValueError(f"missing episode record in {path}")
    return method, episode, episodes[0]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def audit(root: Path, prefix: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit directory: {output}")
    result_root = root / "outputs" / "dynanav_closed_loop"
    files = sorted(result_root.glob(f"{prefix}_m[01]_e*/benchmark_metrics.json"))
    if not files:
        raise FileNotFoundError(f"no benchmark artifacts below {result_root}")

    output.mkdir(parents=True)
    graph_dir = output / "pmg_graphs"
    gpu_dir = output / "gpu_memory"
    graph_dir.mkdir()
    gpu_dir.mkdir()

    records: dict[str, dict[int, dict[str, Any]]] = {"M0": {}, "M1": {}}
    manifest_rows: list[dict[str, Any]] = []
    for artifact in files:
        method, episode, record = episode_from_file(artifact)
        trace = record.get("action_trace") if isinstance(record.get("action_trace"), dict) else {}
        trace_exists, trace_lines, trace_hash = trace_stats(trace.get("path"))
        pmg = record.get("pmg") if isinstance(record.get("pmg"), dict) else {}
        run_id = artifact.parent.name
        manifest_row = {
            "run_id": run_id,
            "method": method,
            "episode": episode,
            "artifact_path": str(artifact),
            "artifact_status": record.get("artifact_status"),
            "episode_seed": record.get("episode_seed"),
            "success": bool(record.get("success")),
            "spl": record.get("spl"),
            "navigation_error_m": record.get("navigation_error"),
            "goal_pose_error_m": pmg.get("goal_pose_error_m"),
            "override_count": trace.get("pmg_override_count", 0),
            "action_output_count": trace.get("action_output_count", 0),
            "override_fraction": trace.get("pmg_override_fraction", 0.0),
            "graph_nodes": pmg.get("graph_nodes"),
            "graph_edges": pmg.get("graph_edges"),
            "trace_path": trace.get("path"),
            "trace_exists": trace_exists,
            "trace_lines": trace_lines,
            "trace_sha256": trace_hash,
            "peak_gpu_memory_mib": record.get("peak_gpu_memory_mib"),
            "official_num_people": record.get("official_num_people"),
            "runtime_num_people": record.get("runtime_num_people"),
            "strict_parity_contract": record.get("strict_parity_contract"),
        }
        manifest_rows.append(manifest_row)
        records[method][episode] = record

        gpu_path = gpu_dir / f"{run_id}_gpu_memory_summary.csv"
        write_csv(
            gpu_path,
            [{
                "run_id": run_id,
                "episode": episode,
                "method": method,
                "peak_gpu_memory_mib": record.get("peak_gpu_memory_mib"),
                "source_benchmark": str(artifact),
                "measurement_type": "benchmark_reported_peak",
            }],
            ["run_id", "episode", "method", "peak_gpu_memory_mib", "source_benchmark", "measurement_type"],
        )

        if method == "M1":
            graph_path = graph_dir / f"{run_id}_pmg_graph.json"
            graph_payload = {
                "run_id": run_id,
                "episode": episode,
                "method": method,
                "episode_seed": record.get("episode_seed"),
                "source_benchmark": str(artifact),
                "graph_mode": pmg.get("graph_mode"),
                "graph_nodes": pmg.get("graph_nodes"),
                "graph_edges": pmg.get("graph_edges"),
                "graph_precision": pmg.get("graph_precision"),
                "graph_recall": pmg.get("graph_recall"),
                "graph_catalog_recall": pmg.get("graph_catalog_recall"),
                "retained_node_ids": pmg.get("retained_node_ids", []),
                "retained_node_labels": pmg.get("retained_node_labels", []),
                "retained_relation_keys": pmg.get("retained_relation_keys", []),
                "constraint_satisfaction": pmg.get("constraint_satisfaction"),
                "constraint_satisfaction_by_relation": pmg.get("constraint_satisfaction_by_relation", {}),
                "control_candidate_position": pmg.get("control_candidate_position"),
                "goal_pose_error_m": pmg.get("goal_pose_error_m"),
                "route_candidate_goal_pose_error_m": pmg.get("route_candidate_goal_pose_error_m"),
                "control_policy": pmg.get("control_policy"),
                "action_dataflow": pmg.get("action_dataflow"),
                "association_reason": pmg.get("association_reason"),
                "association_target_phrase": pmg.get("association_target_phrase"),
                "route_last_reason": pmg.get("route_last_reason"),
                "route_reachable": pmg.get("route_reachable"),
            }
            graph_path.write_text(json.dumps(_json_value(graph_payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    paired_rows: list[dict[str, Any]] = []
    common = sorted(set(records["M0"]) & set(records["M1"]))
    for episode in common:
        m0 = records["M0"][episode]
        m1 = records["M1"][episode]
        m1_trace = m1.get("action_trace") if isinstance(m1.get("action_trace"), dict) else {}
        paired_rows.append({
            "episode": episode,
            "m0_seed": m0.get("episode_seed"),
            "m1_seed": m1.get("episode_seed"),
            "seed_match": m0.get("episode_seed") == m1.get("episode_seed"),
            "instruction_match": m0.get("instruction") == m1.get("instruction"),
            "start_match": m0.get("start") == m1.get("start"),
            "goal_match": m0.get("goal") == m1.get("goal"),
            "m0_success": bool(m0.get("success")),
            "m1_success": bool(m1.get("success")),
            "m0_spl": m0.get("spl"),
            "m1_spl": m1.get("spl"),
            "m0_navigation_error_m": m0.get("navigation_error"),
            "m1_navigation_error_m": m1.get("navigation_error"),
            "delta_success": int(bool(m1.get("success"))) - int(bool(m0.get("success"))),
            "delta_spl": (m1.get("spl") or 0.0) - (m0.get("spl") or 0.0),
            "delta_navigation_error_m": (m1.get("navigation_error") or 0.0) - (m0.get("navigation_error") or 0.0),
            "m1_override_fraction": m1_trace.get("pmg_override_fraction"),
            "m1_m0_policy_prefix_count": (m1_trace.get("control_source_counts") or {}).get("m0_policy_prefix", 0),
        })

    manifest_fields = list(manifest_rows[0].keys())
    write_csv(output / "artifact_manifest.csv", manifest_rows, manifest_fields)
    paired_fields = list(paired_rows[0].keys()) if paired_rows else ["episode"]
    write_csv(output / "paired_audit.csv", paired_rows, paired_fields)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "prefix": prefix,
        "artifact_count": len(manifest_rows),
        "m0_count": sum(row["method"] == "M0" for row in manifest_rows),
        "m1_count": sum(row["method"] == "M1" for row in manifest_rows),
        "paired_count": len(paired_rows),
        "trace_missing_count": sum(not row["trace_exists"] for row in manifest_rows),
        "seed_mismatch_count": sum(not row["seed_match"] for row in paired_rows),
        "instruction_mismatch_count": sum(not row["instruction_match"] for row in paired_rows),
        "start_mismatch_count": sum(not row["start_match"] for row in paired_rows),
        "goal_mismatch_count": sum(not row["goal_match"] for row in paired_rows),
        "artifact_manifest": str(output / "artifact_manifest.csv"),
        "paired_audit": str(output / "paired_audit.csv"),
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Strict queue artifact audit",
        "",
        f"- Prefix: `{prefix}`",
        f"- Artifact rows: {summary['artifact_count']} (M0={summary['m0_count']}, M1={summary['m1_count']})",
        f"- Paired rows: {summary['paired_count']}",
        f"- Missing traces: {summary['trace_missing_count']}",
        f"- Seed/instruction/start/goal mismatches: {summary['seed_mismatch_count']}/{summary['instruction_mismatch_count']}/{summary['start_mismatch_count']}/{summary['goal_mismatch_count']}",
        "",
        "All graph and GPU files in this directory are derived audit records; the source benchmark JSON and action trace paths are retained in the CSV manifest.",
    ]
    (output / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(args.root, args.prefix, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
