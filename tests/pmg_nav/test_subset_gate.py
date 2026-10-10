from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pmg_nav"
sys.path.insert(0, str(SCRIPT_DIR))

from subset_utils import (  # noqa: E402
    BenchmarkFormatError,
    _minimal_benchmark_yaml,
    build_records,
    evaluate_subset_gate,
    load_manifest,
    make_manifest,
    manifest_episode_indices,
    select_records,
)


def _episode(index: int, scene: str, people: int, start_x: float = 0.0) -> dict:
    return {
        "scene": scene,
        "start": [start_x, 0.0, 0.01],
        "start_yaw": 0.0,
        "goal": [20.0, 0.0, 0.01],
        "instruction": f"Stop at landmark {index}.",
        "timeout": 40,
        "num_people": people,
        "robot_type": "nova_carter",
    }


def _source() -> dict:
    source = {"seed": 666, "navigation_method": "ticvla", "success_threshold": 1.5}
    # Four route families, two scenes, and all people strata.  Duplicate
    # geometry variants exercise the route-family merge behavior.
    index = 1
    for scene in ("hospital.usd", "office.usd"):
        for route_start in (0.0, 40.0):
            for people in (0, 10, 20):
                source[f"episode_{index}"] = _episode(index, scene, people, route_start)
                index += 1
    return source


def test_selection_is_deterministic_and_covers_strata() -> None:
    source = _source()
    records = build_records(source, seed=17)
    first = select_records(records, 8)
    second = select_records(records, 8)
    assert [record.key for record in first] == [record.key for record in second]
    assert len(first) == 8
    assert {record.scene for record in first} == {"hospital.usd", "office.usd"}
    assert {record.official_num_people for record in first} == {0, 10, 20}
    assert len({record.route_family for record in first}) == 4


def test_route_family_merges_small_start_perturbation() -> None:
    source = {
        "episode_1": _episode(1, "hospital.usd", 0, 0.0),
        "episode_2": _episode(2, "hospital.usd", 10, 1.0),
    }
    records = build_records(source, seed=1)
    assert records[0].route_family == records[1].route_family


def test_manifest_round_trip_and_input_hash(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.yaml"
    # Hashing only needs the exact source bytes; test manifest construction
    # with a compact JSON-compatible source to avoid a parser dependency here.
    benchmark.write_text("seed: 666\n", encoding="utf-8")
    manifest = make_manifest(_source(), benchmark_path=benchmark, seed=4, size=5, methods=("M0", "M2"))
    path = tmp_path / "subset.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_manifest(path)
    assert loaded["benchmark_sha256"] == manifest["benchmark_sha256"]
    assert manifest_episode_indices(loaded) == [int(key.rsplit("_", 1)[1]) for key in loaded["episode_keys"]]

    loaded["episode_keys"].append(loaded["episode_keys"][0])
    path.write_text(json.dumps(loaded), encoding="utf-8")
    with pytest.raises(BenchmarkFormatError):
        load_manifest(path)


def test_subset_gate_requires_complete_paired_rows() -> None:
    manifest = {
        "kind": "pmg_nav_subset_manifest",
        "episode_keys": ["episode_1", "episode_2"],
        "methods": ["M0", "M2"],
    }
    result = {
        "complete": True,
        "methods": {
            "M0": {"success_rate": 0.5, "mean_spl": 0.4, "mean_navigation_error_m": 2.0},
            "M2": {"success_rate": 0.6, "mean_spl": 0.45, "mean_navigation_error_m": 1.5},
        },
        "rows": [
            {"method": "M0", "episode": "episode_1"},
            {"method": "M0", "episode": "episode_2"},
            {"method": "M2", "episode": "episode_1"},
            {"method": "M2", "episode": "episode_2"},
        ],
    }
    gate = evaluate_subset_gate(manifest, result)
    assert gate["passed"] is True
    assert gate["checks"]["metric_deltas"]["success_rate"] == pytest.approx(0.1)

    result["rows"].pop()
    failed = evaluate_subset_gate(manifest, result)
    assert failed["passed"] is False
    assert any("missing" in reason for reason in failed["reasons"])

    result["rows"].append({"method": "M0", "episode": "episode_1"})
    result["rows"].append({"method": "M9", "episode": "episode_1"})
    duplicated = evaluate_subset_gate(manifest, result)
    assert duplicated["passed"] is False
    assert duplicated["checks"]["duplicate_method_episode_pairs"] == ["M0:episode_1"]
    assert duplicated["checks"]["unexpected_method_episode_pairs"] == ["M9:episode_1"]

    incomplete_flag = dict(result)
    incomplete_flag["rows"] = [
        {"method": "M0", "episode": "episode_1"},
        {"method": "M0", "episode": "episode_2"},
        {"method": "M2", "episode": "episode_1"},
        {"method": "M2", "episode": "episode_2"},
    ]
    incomplete_flag.pop("complete")
    rejected = evaluate_subset_gate(manifest, incomplete_flag)
    assert rejected["passed"] is False
    assert any("complete=true" in reason for reason in rejected["reasons"])


def test_minimal_yaml_fallback_accepts_benchmark_shape() -> None:
    parsed = _minimal_benchmark_yaml(
        """# comment
seed: 666
episode_1:
  scene: hospital.usd
  start:
  - -3.0
  - -1.2
  - 0.01
  start_yaw: 0.0
  goal:
  - 9.52
  - 2.33
  - 0.01
  instruction: Stop at the vending machine.
  timeout: 40
  num_people: 0
  robot_type: nova_carter
"""
    )
    assert parsed["seed"] == 666
    assert parsed["episode_1"]["start"] == [-3.0, -1.2, 0.01]
    assert parsed["episode_1"]["num_people"] == 0
