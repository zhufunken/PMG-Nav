#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    episodes = result.get("episodes", [])
    artifact_path = args.result.parent / "telemetry" / "artifacts.json"
    artifacts = json.loads(artifact_path.read_text(encoding="utf-8"))

    latencies = [float(value) for value in re.findall(r"latency=([0-9.]+)s", log_text)]
    model_actions = re.findall(r"v_cmd=([-+0-9.eE]+),\s*w_cmd=([-+0-9.eE]+)", log_text)
    gpu_samples = []
    with args.gpu_log.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                gpu_samples.append(float(row["memory_used_mib"]))
            except (KeyError, TypeError, ValueError):
                continue

    normal_episodes = []
    for episode in episodes:
        duration = float(episode.get("duration", 0.0) or 0.0)
        frames = int(episode.get("duration_frames", 0) or 0)
        error = episode.get("error")
        artifact_status = episode.get("artifact_status", "valid")
        normal = not error and artifact_status == "valid" and duration > 0 and frames > 0
        normal_episodes.append(
            {
                "episode": episode.get("episode"),
                "normal_termination": normal,
                "success": bool(episode.get("success", False)),
                "duration_seconds": duration,
                "duration_frames": frames,
                "fps": frames / duration if duration > 0 else 0.0,
                "navigation_error_m": episode.get("navigation_error"),
                "spl": episode.get("spl"),
                "collision": episode.get("collision"),
                "human_collision": episode.get("human_collision"),
                "physical_collision": episode.get("physical_collision"),
                "error": error,
                "artifact_status": artifact_status,
                "artifact_invalid_reasons": episode.get("artifact_invalid_reasons", []),
            }
        )

    metrics = {
        "status": "passed" if normal_episodes and all(item["normal_termination"] for item in normal_episodes) else "failed",
        "total_episodes": len(normal_episodes),
        "normal_episodes": sum(item["normal_termination"] for item in normal_episodes),
        "successful_episodes": sum(item["success"] for item in normal_episodes),
        "mean_fps": (
            sum(item["fps"] for item in normal_episodes) / len(normal_episodes)
            if normal_episodes else 0.0
        ),
        "mean_vlm_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "vlm_latency_observations": len(latencies),
        "model_action_observations": len(model_actions),
        "peak_gpu_memory_mib": max(gpu_samples) if gpu_samples else None,
        "average_metrics": result.get("average_metrics", {}),
        "episodes": normal_episodes,
        "artifacts": artifacts,
        "raw_result": str(args.result),
        "log": str(args.log),
        "gpu_memory_log": str(args.gpu_log),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    if metrics["status"] != "passed":
        raise SystemExit(1)
    if artifacts.get("status") != "passed" or artifacts.get("rgb_frames", 0) <= 0:
        raise SystemExit(2)
    if artifacts.get("trajectory_points", 0) <= 0:
        raise SystemExit(3)
    if not math.isfinite(metrics["mean_fps"]) or metrics["mean_fps"] <= 0:
        raise SystemExit(4)
    if metrics["vlm_latency_observations"] <= 0 or metrics["model_action_observations"] <= 0:
        raise SystemExit(5)
    if artifacts.get("nonzero_command_points", 0) <= 0:
        raise SystemExit(6)


if __name__ == "__main__":
    main()
