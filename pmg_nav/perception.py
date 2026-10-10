"""Stage 2 perception interfaces.

`SceneOraclePerception` is the annotation-gated implementation used on the
current DynaNav USDs. The interface is intentionally detector-shaped so a
GroundingDINO/RGB-D backend can be swapped in without changing PMG updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Detection3D:
    node_id: str
    label: str
    position: tuple[float, float, float]
    confidence: float
    frame_id: int
    source: str
    bbox_2d: tuple[float, float, float, float] | None = None
    depth_m: float | None = None


class SceneOraclePerception:
    source = "isaac_usd_oracle_rgbd"

    def __init__(self, max_range_m: float = 35.0) -> None:
        self.max_range_m = max_range_m
        self.catalog: list[Detection3D] = []

    def reset(self) -> None:
        self.catalog.clear()

    def detections(self, robot_position: tuple[float, float, float], frame_id: int) -> list[Detection3D]:
        detections = []
        for item in self.catalog:
            dx = item.position[0] - robot_position[0]
            dy = item.position[1] - robot_position[1]
            distance = (dx * dx + dy * dy + (item.position[2] - robot_position[2]) ** 2) ** 0.5
            if distance <= self.max_range_m:
                confidence = max(0.35, min(0.99, 1.0 - distance / (self.max_range_m * 1.4)))
                detections.append(Detection3D(item.node_id, item.label, item.position, confidence, frame_id, self.source, depth_m=distance))
        return detections

    def add_scene_catalog(self, items: list[dict[str, Any]]) -> None:
        self.catalog = [
            Detection3D(str(item["node_id"]), str(item["label"]), tuple(item["position"]), 1.0, 0, self.source)
            for item in items
        ]


class PredictedPerception:
    """Detector/RGB-D shaped prediction backend for diagnostic runs.

    The compatibility runner does not fabricate detector output.  A prediction
    file must be supplied explicitly (JSON list or ``{"frames": ...}``) so a
    run can be reproduced and audited.  Missing input raises immediately,
    preventing an accidental oracle fallback for M2.
    """

    source = "predicted_rgbd"

    def __init__(self, predictions_path: str | Path | None) -> None:
        if not predictions_path:
            raise RuntimeError(
                "M2 blocked: PMG_PREDICTED_DETECTIONS_PATH is not configured; "
                "no GroundingDINO/RGB-D prediction backend is available"
            )
        path = Path(predictions_path)
        if not path.is_file():
            raise RuntimeError(f"M2 blocked: predicted detections file is missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("frames", payload.get("detections", []))
        if not isinstance(payload, list):
            raise ValueError("predicted detections must be a JSON list or a frames/detections object")
        self._frames: dict[int, list[Detection3D]] = {}
        for frame in payload:
            if not isinstance(frame, dict):
                continue
            frame_id = int(frame.get("frame_id", 0))
            entries = frame.get("detections", [frame])
            if isinstance(entries, dict):
                entries = [entries]
            parsed: list[Detection3D] = []
            for item in entries:
                if not isinstance(item, dict) or "position" not in item or "label" not in item:
                    continue
                position = tuple(float(value) for value in item["position"])
                if len(position) != 3:
                    continue
                parsed.append(
                    Detection3D(
                        str(item.get("node_id", f"pred:{frame_id}:{len(parsed)}")),
                        str(item["label"]),
                        position,
                        float(item.get("confidence", 0.5)),
                        frame_id,
                        self.source,
                        tuple(item["bbox_2d"]) if item.get("bbox_2d") else None,
                        float(item["depth_m"]) if item.get("depth_m") is not None else None,
                    )
                )
            self._frames[frame_id] = parsed

    def detections(self, robot_position: tuple[float, float, float], frame_id: int) -> list[Detection3D]:
        del robot_position
        return list(self._frames.get(frame_id, self._frames.get(0, [])))
