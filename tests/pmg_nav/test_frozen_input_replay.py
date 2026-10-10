from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pmg_nav"
sys.path.insert(0, str(SCRIPT_DIR))

from frozen_input_replay import (  # noqa: E402
    _action_checksum,
    build_m3_shadow,
    replay,
    validate_shadow,
)


def _payload() -> dict:
    return {
        "instruction": "Go to the chair.",
        "image_refs": ["front_frame_000003.jpg"],
        "image_sha256": [""],
        "robot_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "robot_pose": {
            "position": [0.0, 0.0, 0.0],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
        },
        "delay_time": 0.0,
        "robot_type": "wheeled robot",
        "inference_step": 3,
    }


def _rows() -> list[dict]:
    output = {
        "kind": "output",
        "mode": "M0",
        "simulation_frame": 3,
        "v_cmd": 1.5,
        "w_cmd": 0.25,
        "pmg_override": False,
    }
    output["checksum"] = _action_checksum(output)
    return [
        {
            "kind": "input",
            "mode": "M0",
            "simulation_frame": 3,
            "inference_step": 3,
            "policy_input_checksum": "input-checksum",
            "policy_input_payload": _payload(),
        },
        {
            "kind": "inference",
            "mode": "M0",
            "simulation_frame": 3,
            "inference_step": 3,
        },
        output,
    ]


def test_frozen_shadow_preserves_payload_and_action_checksum(tmp_path: Path) -> None:
    source = tmp_path / "m0.jsonl"
    shadow = tmp_path / "m3_shadow.jsonl"
    report = tmp_path / "report.json"
    source.write_text(
        "\n".join(json.dumps(row) for row in _rows()) + "\n", encoding="utf-8"
    )

    result = replay(source, shadow, report)

    assert result["status"] == "passed"
    assert result["execution"]["isaac_started"] is False
    assert result["execution"]["model_loaded"] is False
    assert result["same_input"]["policy_input_payload_exact_copy"] is True
    assert result["action_checksum"]["recorded_checksum_matches_source"] is True
    shadow_rows = [json.loads(line) for line in shadow.read_text().splitlines()]
    assert all(row["mode"] == "M3" for row in shadow_rows)
    assert shadow_rows[0]["policy_input_payload"] == _payload()
    assert shadow_rows[-1]["checksum"] == _rows()[-1]["checksum"]
    assert json.loads(report.read_text())["kind"] == "stage2_frozen_input_m3_shadow"


def test_frozen_shadow_reports_image_hash_evidence(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    image = image_root / "front_frame_000003.jpg"
    image_bytes = b"frozen-image"
    image.write_bytes(image_bytes)
    rows = _rows()
    rows[0]["policy_input_payload"]["image_sha256"] = [
        hashlib.sha256(image_bytes).hexdigest()
    ]
    result = validate_shadow(
        rows,
        build_m3_shadow(rows),
        image_root=image_root,
    )
    assert result["status"] == "passed"
    assert result["image_evidence"]["content_hash_checked_count"] == 1
    assert result["image_evidence"]["content_hash_match_count"] == 1


def test_frozen_shadow_blocks_missing_payload_or_non_m0_source() -> None:
    rows = _rows()
    rows[0].pop("policy_input_payload")
    result = validate_shadow(rows, build_m3_shadow(rows))
    assert result["status"] == "blocked_or_failed"
    assert result["same_input"]["policy_input_payload_complete"] is False

    rows = _rows()
    rows[0]["mode"] = "M1"
    result = validate_shadow(rows, build_m3_shadow(rows))
    assert result["status"] == "blocked_or_failed"
    assert result["source"]["source_is_m0_like"] is False


def test_frozen_shadow_detects_tampered_action() -> None:
    rows = _rows()
    shadow = build_m3_shadow(rows)
    shadow[-1]["w_cmd"] = 0.5
    result = validate_shadow(rows, shadow)
    assert result["status"] == "blocked_or_failed"
    assert result["action_checksum"]["first_command_mismatch_index"] == 0
