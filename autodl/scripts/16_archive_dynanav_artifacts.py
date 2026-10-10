#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    return parser.parse_args()


def write_video(frame_paths: list[Path], output: Path, fps: float) -> int:
    first = cv2.imread(str(frame_paths[0]))
    if first is None or first.size == 0:
        raise RuntimeError(f"first RGB frame is unreadable: {frame_paths[0]}")

    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize the MP4 writer")

    written = 0
    try:
        for path in frame_paths:
            frame = cv2.imread(str(path))
            if frame is None or frame.size == 0:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    if written == 0 or not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("no non-empty RGB video was produced")
    return written


def write_trajectory(state_paths: list[Path], output: Path) -> tuple[int, int]:
    fields = [
        "frame_count",
        "x",
        "y",
        "z",
        "qw",
        "qx",
        "qy",
        "qz",
        "vx",
        "vy",
        "vz",
        "yaw_speed",
        "command_v",
        "command_w",
    ]
    rows = []
    for path in state_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        position = list(payload.get("position", [])) + [None] * 3
        quaternion = list(payload.get("quaternion", [])) + [None] * 4
        velocity = list(payload.get("velocity", [])) + [None] * 3
        commands = payload.get("commands", {})
        rows.append(
            {
                "frame_count": payload.get("frame_count"),
                "x": position[0],
                "y": position[1],
                "z": position[2],
                "qw": quaternion[0],
                "qx": quaternion[1],
                "qy": quaternion[2],
                "qz": quaternion[3],
                "vx": velocity[0],
                "vy": velocity[1],
                "vz": velocity[2],
                "yaw_speed": payload.get("yaw_speed"),
                "command_v": commands.get("v"),
                "command_w": commands.get("w"),
            }
        )

    if not rows:
        raise RuntimeError("no robot-state telemetry was produced")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    nonzero_commands = sum(
        abs(float(row["command_v"] or 0.0)) > 1e-4
        or abs(float(row["command_w"] or 0.0)) > 1e-4
        for row in rows
    )
    return len(rows), nonzero_commands


def main() -> None:
    args = parse_args()
    frame_paths = sorted(args.telemetry_dir.glob("tp_frame_*.jpg"))
    state_paths = sorted(args.telemetry_dir.glob("robot_state_frame_*.json"))
    if not frame_paths:
        raise FileNotFoundError(f"no third-person RGB frames in {args.telemetry_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "episode.mp4"
    trajectory_path = args.output_dir / "trajectory.csv"
    first_rgb = args.output_dir / "first_rgb.jpg"
    last_rgb = args.output_dir / "last_rgb.jpg"

    skip_video = os.environ.get("PMG_SKIP_VIDEO", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    video_frames = 0 if skip_video else write_video(frame_paths, video_path, args.fps)
    trajectory_points, nonzero_command_points = write_trajectory(state_paths, trajectory_path)
    shutil.copy2(frame_paths[0], first_rgb)
    shutil.copy2(frame_paths[-1], last_rgb)

    manifest = {
        "status": "passed",
        "rgb_frames": len(frame_paths),
        "rgb_width": int(cv2.imread(str(frame_paths[0])).shape[1]),
        "rgb_height": int(cv2.imread(str(frame_paths[0])).shape[0]),
        "video_fps": args.fps,
        "video_seconds": video_frames / args.fps,
        "video_path": None if skip_video else str(video_path),
        "video_bytes": 0 if skip_video else video_path.stat().st_size,
        "video_skipped": skip_video,
        "trajectory_points": trajectory_points,
        "nonzero_command_points": nonzero_command_points,
        "trajectory_path": str(trajectory_path),
        "first_rgb_path": str(first_rgb),
        "last_rgb_path": str(last_rgb),
        "telemetry_source": str(args.telemetry_dir),
    }
    manifest_path = args.output_dir / "artifacts.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
