from __future__ import annotations

import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pmg_nav"
sys.path.insert(0, str(SCRIPT_DIR))

from check_observation_parity import check_observations  # noqa: E402


def _write_trace(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "kind": "input",
                "simulation_frame": 3,
                "inference_step": 3,
                "policy_input_payload": payload,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _payload() -> dict:
    return {
        "instruction": "go",
        "image_refs": ["front_frame_000003.jpg"],
        "image_sha256": ["a" * 64],
        "robot_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "robot_pose": {"position": [0.0, 0.0, 0.0], "quaternion": [1.0, 0.0, 0.0, 0.0]},
        "delay_time": 0.0,
        "robot_type": "wheeled robot",
        "inference_step": 3,
    }


def test_observation_parity_requires_content_and_state(tmp_path: Path) -> None:
    baseline = tmp_path / "m0.jsonl"
    candidate = tmp_path / "m3.jsonl"
    payload = _payload()
    _write_trace(baseline, payload)
    _write_trace(candidate, payload)
    result = check_observations(baseline, candidate)
    assert result["status"] == "passed"

    changed = dict(payload)
    changed["image_sha256"] = ["b" * 64]
    _write_trace(candidate, changed)
    result = check_observations(baseline, candidate)
    assert result["status"] == "failed"
    assert result["first_observation_payload_mismatch_index"] == 0
    assert result["first_field_mismatch_index"]["image_sha256"] == 0


def test_observation_parity_rejects_legacy_payload(tmp_path: Path) -> None:
    baseline = tmp_path / "m0.jsonl"
    candidate = tmp_path / "m3.jsonl"
    legacy = {"image_refs": ["front_frame_000003.jpg"], "delay_time": 0.0}
    _write_trace(baseline, legacy)
    _write_trace(candidate, legacy)
    result = check_observations(baseline, candidate)
    assert result["status"] == "failed"
    assert result["observation_payload_schema_available"] is False
