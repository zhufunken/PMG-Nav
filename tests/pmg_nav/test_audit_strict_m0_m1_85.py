from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "pmg_nav"
    / "audit_strict_m0_m1_85.py"
)
SPEC = importlib.util.spec_from_file_location("strict_queue_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _output_row(*, override: bool) -> dict:
    """Return a complete strict action output record."""

    return {
        "kind": "output",
        "policy_v_cmd": 1.0,
        "policy_w_cmd": 0.1,
        "v_cmd": 0.8 if override else 1.0,
        "w_cmd": 0.2 if override else 0.1,
        "pmg_override": override,
        "pmg_override_fraction": 0.5,
        "pmg_control_source": "pmg" if override else "policy",
        "pmg_gate_scope": "global" if override else "policy",
        "pmg_gate_reason": "goal_candidate" if override else "not_terminal",
        "pmg_route_valid": True,
        "pmg_waypoint_index": 0,
        "pmg_route_alignment_hold": False,
        "pmg_route_stall_policy_fallback": False,
        "pmg_semantic_candidate": {"position": [1.0, 2.0, 0.0]},
        "pmg_control_candidate": {"position": [1.0, 2.0, 0.0]},
    }


def _write_strict_trace(path: Path, *, override: bool = True) -> None:
    rows = [
        {"kind": "input", "simulation_frame": 0},
        {"kind": "inference", "simulation_frame": 0},
        _output_row(override=override),
        _output_row(override=False),
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _record(episode: int, method: str) -> dict:
    return {
        "episode": f"episode_{episode}",
        "run_id": f"{method}_episode_{episode:03d}",
        "episode_seed": 666 + episode,
        "instruction": "Stop near the blue chair.",
        "start": [0.0, 0.0, 0.01],
        "goal": [4.0, 2.0, 0.01],
        "success": True,
        "spl": 0.75,
        "navigation_error": 0.35,
        "artifact_status": "valid",
        "official_num_people": 3,
        "runtime_num_people": 0,
        "peak_gpu_memory_mib": 110.0,
        "gpu_memory_csv": "gpu_memory.csv",
        "action_trace": {
            "path": "action_trace.jsonl",
            "entries": 4,
            "action_output_count": 2,
            "action_input_count": 1,
            "inference_event_count": 1,
            "pmg_override_count": 1 if method == "M1" else 0,
            "pmg_override_fraction": 0.5 if method == "M1" else 0.0,
        },
        "control_gate": {
            "pmg_override_count": 1 if method == "M1" else 0,
            "pmg_override_fraction": 0.5 if method == "M1" else 0.0,
            "action_output_count": 2,
        },
        "pmg": {
            "graph_path": "episode_1.graph.json",
            "graph_mode": "compact",
            "graph_nodes": 2,
            "graph_edges": 2,
            "max_graph_nodes": 4,
            "max_edges_per_anchor": 2,
            "goal_pose_error_m": 0.25,
            "semantic_goal_pose_error_m": 0.25,
            "control_candidate_position": [1.0, 2.0, 0.0],
            "control_candidate_source": "goal_field",
            "pinned_anchor_retained": True,
        },
    }


def _write_run(root: Path, method: str, *, instruction: str | None = None) -> Path:
    run = root / f"method_{method.lower()}" / "episode_001"
    (run / "telemetry").mkdir(parents=True)
    record = _record(1, method)
    if instruction is not None:
        record["instruction"] = instruction
    (run / "benchmark_metrics.json").write_text(
        json.dumps({"episodes": [record]}, indent=2) + "\n", encoding="utf-8"
    )
    _write_strict_trace(run / "action_trace.jsonl", override=method == "M1")
    (run / "telemetry" / "artifacts.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "rgb_frames": 4,
                "trajectory_points": 4,
                "nonzero_command_points": 3,
                "rgb_width": 320,
                "rgb_height": 240,
            }
        ),
        encoding="utf-8",
    )
    (run / "gpu_memory.csv").write_text(
        "epoch,memory_used_mib\n0,100\n1,110\n", encoding="utf-8"
    )
    (run / "episode_1.graph.json").write_text(
        json.dumps(
            {
                "version": 1,
                "association_radius": 0.75,
                "objects": {"chair": {}, "desk": {}},
                "relations": {"chair|near|desk": {}, "chair|left|desk": {}},
            }
        ),
        encoding="utf-8",
    )
    return run


def _paired_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "queue"
    m0 = _write_run(root, "M0")
    m1 = _write_run(root, "M1")
    return root, m0, m1


class StrictQueueAuditTests(unittest.TestCase):
    def test_complete_pair_validates_artifacts_and_writes_derived_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            root, _, _ = _paired_fixture(tmp_path)

            report, derived_graphs, rows = AUDIT.audit_queue(root, expected_episodes=[1])

            self.assertTrue(report["complete"])
            self.assertEqual(report["valid_rows"], 2)
            self.assertEqual(len(derived_graphs), 2)
            self.assertTrue(all(row["status"] == "pass" for row in rows))
            m1 = next(row for row in rows if row["method"] == "M1")
            self.assertEqual(m1["action_trace"]["status"], "pass")
            self.assertEqual(m1["action_trace"]["override_count"], 1)
            self.assertEqual(m1["override"]["fraction"], 0.5)
            self.assertEqual(m1["pmg_graph"]["status"], "pass")
            self.assertEqual(m1["pmg_graph"]["actual_graph_nodes"], 2)
            self.assertEqual(m1["gpu_memory"]["peak_mib"], 110.0)
            self.assertTrue(report["pairs"][0]["parity"]["all_equal"])

            output_dir = tmp_path / "audit-output"
            AUDIT.write_outputs(report, derived_graphs, rows, output_dir)
            self.assertTrue((output_dir / "audit_manifest.json").is_file())
            self.assertTrue((output_dir / "pmg_graphs" / "M1_episode_1.json").is_file())
            summary = (output_dir / "gpu_peak_summary.csv").read_text(encoding="utf-8")
            self.assertIn("M1,episode_1", summary)
            self.assertIn(",110.0,110.0,2,", summary)

    def test_pair_parity_mismatch_is_explicit_and_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _, m1 = _paired_fixture(Path(temporary))
            metrics_path = m1 / "benchmark_metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            payload["episodes"][0]["goal"] = [4.5, 2.0, 0.01]
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            report, _, rows = AUDIT.audit_queue(root, expected_episodes=[1])

            self.assertFalse(report["complete"])
            self.assertTrue(all(row["status"] == "pass" for row in rows))
            parity = report["pairs"][0]["parity"]
            self.assertFalse(parity["all_equal"])
            self.assertEqual(parity["mismatches"], ["goal"])
            reasons = {item["reason"] for item in report["failures"] if isinstance(item, dict)}
            self.assertIn("parity:goal", reasons)

    def test_missing_trace_remains_visible_as_row_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _, m1 = _paired_fixture(Path(temporary))
            (m1 / "action_trace.jsonl").unlink()

            report, _, rows = AUDIT.audit_queue(root, expected_episodes=[1])

            self.assertFalse(report["complete"])
            m1_row = next(row for row in rows if row["method"] == "M1")
            self.assertEqual(m1_row["status"], "fail")
            self.assertIn("action_trace:action_trace_missing", m1_row["failures"])
            self.assertEqual(m1_row["override"]["status"], "pass")

    def test_write_outputs_refuses_to_overwrite_nonempty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            root, _, _ = _paired_fixture(tmp_path)
            report, derived_graphs, rows = AUDIT.audit_queue(root, expected_episodes=[1])
            output_dir = tmp_path / "existing-audit"
            output_dir.mkdir()
            sentinel = output_dir / "sentinel.txt"
            sentinel.write_text("preserve", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                AUDIT.write_outputs(report, derived_graphs, rows, output_dir)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

            empty_output_dir = tmp_path / "existing-empty-audit"
            empty_output_dir.mkdir()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                AUDIT.write_outputs(report, derived_graphs, rows, empty_output_dir)
