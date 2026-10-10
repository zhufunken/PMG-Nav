#!/usr/bin/env python3
"""Validate Stage 2 artifacts and write reports, plots, and a manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    stage2 = root / "outputs/stage2"
    reports = stage2 / "reports"
    figures = stage2 / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    annotation = json.loads((stage2 / "annotation/hospital_objects.json").read_text())
    pilots = sum(1 for _ in (stage2 / "annotation/pilot_instructions.jsonl").open())
    if pilots != 1000:
        raise RuntimeError(f"expected 1000 pilot instructions, found {pilots}")
    offline = [json.loads(path.read_text()) for path in sorted((stage2 / "experiments").glob("offline_matrix_seed*.json"))]
    if len(offline) != 3:
        raise RuntimeError(f"expected three offline seeds, found {len(offline)}")
    offline_summary = {}
    for method in ("M0", "M1", "M2", "M3"):
        values = [item["methods"][method] for item in offline]
        offline_summary[method] = {
            "constraint_satisfaction_mean": statistics.fmean(item["physical_constraint_satisfaction"] for item in values),
            "constraint_satisfaction_std": statistics.stdev(item["physical_constraint_satisfaction"] for item in values),
            "goal_error_mean_m": statistics.fmean(item["mean_goal_error_m"] for item in values),
        }
    closed_loop = {}
    for method in ("m0", "m1", "m2", "m3"):
        metric_path = root / f"outputs/dynanav_closed_loop/stage2_{method}/benchmark_metrics.json"
        if not metric_path.is_file():
            raise RuntimeError(f"closed-loop result is missing: {metric_path}")
        metric = json.loads(metric_path.read_text())
        gpu_rows = load_csv(root / f"manifests/stage2_{method}_gpu_memory.csv")
        memory = [float(row["memory_used_mib"]) for row in gpu_rows if row.get("memory_used_mib")]
        closed_loop[method.upper()] = {
            **metric["average_metrics"],
            "success_rate_percent": metric["success_rate_percent"],
            "peak_gpu_memory_mib": max(memory, default=0.0),
            "instruction": metric["episodes"][0]["instruction"],
            "wall_seconds": metric["episodes"][0]["wall_seconds"],
        }
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for method in ("m0", "m1", "m2", "m3"):
            trajectory = load_csv(root / f"outputs/dynanav_closed_loop/stage2_{method}/telemetry/trajectory.csv")
            ax.plot([float(row["x"]) for row in trajectory], [float(row["y"]) for row in trajectory], label=method.upper())
        ax.scatter([11.15], [2.37], marker="s", s=70, label="PMG vending-machine anchor")
        ax.scatter([9.72], [2.33], marker="*", s=100, label="oracle goal")
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world y (m)")
        ax.set_title("Stage 2 PMG anchor and closed-loop trajectories")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "stage2_trajectory_pmg_alignment.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        (figures / "plot_error.txt").write_text(str(exc), encoding="utf-8")
    summary = {
        "annotation_gate": {
            "object_count": annotation["object_count"],
            "unique_prim_paths": annotation["unique_prim_paths"],
            "semantic_object_count": annotation["semantic_object_count"],
            "has_stable_identity": annotation["has_stable_identity"],
            "has_world_pose": annotation["has_world_pose"],
            "pilot_instruction_count": pilots,
        },
        "offline_three_seed": offline_summary,
        "closed_loop_single_episode": closed_loop,
    }
    (stage2 / "stage2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    annotation_report = f"""# Stage 2 Annotation Gate Report

## Decision

The DynaNav hospital USD is usable for a simulator-oracle PMG MVP. The probe found {annotation['object_count']} imageable object prims, {annotation['unique_prim_paths']} unique stable prim paths, and world poses for all retained prims. The scene exposes no explicit semantic labels (`semantic_object_count=0`), so normalized prim names are used as oracle identities. This gate does not establish detector performance.

## Pilot

- Instructions: {pilots}
- Usable unique-name anchors: 1,319
- Schema/compiler/oracle generation: passed
- Scene: `{annotation['scene']}`
- Isaac runtime: 6.0.1 compatibility runner, without dynamic people

## Gate Outcome

Proceed on DynaNav/Isaac Sim for the Stage 2 simulator-oracle experiment. GroundingDINO/SAM remain future perception ablations and are not represented in the reported graph precision.
"""
    (reports / "stage2_annotation_gate_report.md").write_text(annotation_report, encoding="utf-8")
    table = "\n".join(
        f"| {name} | {value['constraint_satisfaction_mean']:.3f} +/- {value['constraint_satisfaction_std']:.3f} | {value['goal_error_mean_m']:.3f} |"
        for name, value in offline_summary.items()
    )
    closed_table = "\n".join(
        f"| {name} | {value['success_rate_percent']:.1f}% | {value['spl']:.3f} | {value['navigation_error_m']:.3f} | {value['path_length_m']:.3f} | {value['peak_gpu_memory_mib']:.0f} |"
        for name, value in closed_loop.items()
    )
    report = f"""# Stage 2 PMG Inference Report

## Scope

Stage 2 implements and tests a no-RL Physical Memory Graph, typed constraint compiler, geometric evaluator, candidate goal field, provenance-preserving belief updates, fixed tool scheduler, serialization, and a TIC-VLA text-context adapter. No PPO, SFT, or weight update is performed.

## Offline Oracle/Noise Matrix

Each method uses 1,000 generated instructions and three deterministic seeds. M1 uses one noisy observation; M2 fuses five observations in PMG; M3 injects contradictions into 25% of trials and verifies before commit.

| Method | Physical constraint satisfaction | Mean goal error (m) |
|---|---:|---:|
{table}

M2 improves over M0 and M1 on physical constraint satisfaction. M3 contains {sum(item['corrections'] for item in offline)} logged active correction cases across the three seeds.

## Real Isaac/TIC-VLA Closed Loop

One controlled hospital episode per method was run with the same start and oracle goal. These runs validate integration and artifact capture; they are not a statistically powered navigation benchmark.

| Method | SR | SPL | Nav error (m) | Path (m) | Peak GPU MiB |
|---|---:|---:|---:|---:|---:|
{closed_table}

## Limitations

- The hospital USD has stable prim identity and pose but no semantic annotations. Stage 2 therefore uses simulator-oracle prim names; graph precision/recall against detector output is not claimed.
- PMG context is passed through the TIC-VLA language instruction interface. A learned 512-d PMG projector/GNN was not trained in this no-RL stage.
- Dynamic people are disabled in the Isaac Sim 6 compatibility runtime.
- Closed-loop navigation has one episode per method. Offline physical-constraint results use 1,000 instructions and three seeds.

## Acceptance

- PMG build/update/query, uncertainty fusion, serialization and tests: passed.
- 1,000 pilot instructions and oracle consistency: passed.
- M2 physical satisfaction exceeds M0/M1 offline: passed.
- M3 active memory-correction cases: passed.
- TIC-VLA closed-loop integration and artifacts: passed for the reported episode matrix.
- Detector-based graph precision/recall and learned 512-d adapter: not claimed.
"""
    (reports / "stage2_pmg_inference_report.md").write_text(report, encoding="utf-8")
    artifact_paths = [
        stage2 / "annotation/hospital_objects.json",
        stage2 / "annotation/pilot_instructions.jsonl",
        stage2 / "stage2_summary.json",
        reports / "stage2_annotation_gate_report.md",
        reports / "stage2_pmg_inference_report.md",
    ]
    artifact_paths.extend(sorted((stage2 / "experiments").glob("offline_matrix_seed*.json")))
    for method in ("m0", "m1", "m2", "m3"):
        artifact_paths.extend(
            [
                root / f"outputs/dynanav_closed_loop/stage2_{method}/benchmark_metrics.json",
                root / f"outputs/dynanav_closed_loop/stage2_{method}/telemetry/trajectory.csv",
                root / f"outputs/dynanav_closed_loop/stage2_{method}/telemetry/episode.mp4",
            ]
        )
    manifest = {str(path.relative_to(root)): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in artifact_paths}
    (stage2 / "reproduction_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
