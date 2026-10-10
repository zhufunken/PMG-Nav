from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "pmg_nav"
    / "report_strict_m0_m1_85.py"
)
SPEC = importlib.util.spec_from_file_location("strict_queue_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


def episode_record(
    episode: int,
    *,
    success: bool,
    spl: float,
    navigation_error: float,
    override_fraction: float = 0.0,
    artifact_status: str = "valid",
    error: str | None = None,
) -> dict:
    return {
        "episode": f"episode_{episode}",
        "episode_seed": 666 + episode,
        "success": success,
        "spl": spl,
        "navigation_error": navigation_error,
        "duration": 40.0,
        "duration_frames": 1200,
        "artifact_status": artifact_status,
        "artifact_invalid_reasons": [] if artifact_status == "valid" else ["runner_error"],
        "error": error,
        "strict_parity_contract": True,
        "parity_fixed_horizon": True,
        "deterministic_torch_cuda": True,
        "runtime_num_people": 0,
        "official_num_people": 3,
        "compatibility_runner": "isaac_sim_6_without_ira_0x",
        "control_gate": {
            "pmg_override_fraction": override_fraction,
            "pmg_override_count": 3,
            "action_output_count": 10,
        },
    }


class StrictQueueReportTests(unittest.TestCase):
    def test_nested_episode_records_pair_and_compute_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m0_dir = root / "method_m0" / "episode_001"
            m1_dir = root / "method_m1" / "episode_001"
            m0_dir.mkdir(parents=True)
            m1_dir.mkdir(parents=True)
            (m0_dir / "benchmark_metrics.json").write_text(
                json.dumps({"episodes": [episode_record(1, success=False, spl=0.0, navigation_error=8.0)]}),
                encoding="utf-8",
            )
            (m1_dir / "benchmark_metrics.json").write_text(
                json.dumps({"episodes": [episode_record(1, success=True, spl=0.6, navigation_error=2.0, override_fraction=0.3)]}),
                encoding="utf-8",
            )

            report = REPORT.aggregate_queue(root, expected_episodes=[1], bootstrap_samples=100)

        self.assertTrue(report["complete"])
        self.assertEqual(report["methods"]["M0"]["sr"], 0.0)
        self.assertEqual(report["methods"]["M1"]["sr"], 1.0)
        self.assertAlmostEqual(report["methods"]["M1"]["override_fraction"], 0.3)
        paired = report["paired_statistics"]
        self.assertEqual(paired["paired_episodes"], 1)
        self.assertEqual(paired["sr"]["mean_delta"], 1.0)
        self.assertAlmostEqual(paired["spl"]["mean_delta"], 0.6)
        self.assertEqual(paired["navigation_error_m"]["mean_delta"], -6.0)
        self.assertIn("runtime_num_people=0", " ".join(report["protocol_caveats"]))

    def test_nested_mapping_schema_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "benchmark_metrics.json"
            result.write_text(
                json.dumps(
                    {
                        "results": {
                            "M0": {
                                "episode_2": {
                                    "metrics": episode_record(2, success=True, spl=0.5, navigation_error=3.0)
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            candidates, failures = REPORT.discover_records(result)

        self.assertEqual(failures, [])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].method, "M0")
        self.assertEqual(candidates[0].episode, "episode_2")
        self.assertEqual(candidates[0].record["spl"], 0.5)

    def test_top_level_protocol_and_caveat_are_carried_to_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "m0" / "benchmark_metrics.json"
            result.parent.mkdir()
            payload = {
                "artifact_status": "valid",
                "strict_parity_contract": True,
                "parity_fixed_horizon": True,
                "deterministic_torch_cuda": True,
                "protocol_caveats": ["fixture caveat"],
                "episodes": [episode_record(1, success=True, spl=0.4, navigation_error=2.0)],
            }
            result.write_text(json.dumps(payload), encoding="utf-8")
            report = REPORT.aggregate_queue(Path(temporary), expected_episodes=[1])

        self.assertTrue(report["complete"] is False)
        self.assertIn("fixture caveat", report["protocol_caveats"])
        row = next(row for row in report["rows"] if row["method"] == "M0")
        self.assertEqual(row["status"], "valid")

    def test_failed_runner_artifact_remains_visible_and_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for method in ("m0", "m1"):
                directory = root / method
                directory.mkdir()
                record = episode_record(
                    1,
                    success=False,
                    spl=0.0,
                    navigation_error=0.0,
                    artifact_status="invalid_artifact" if method == "m1" else "valid",
                    error="compatibility_runner:crash" if method == "m1" else None,
                )
                (directory / "benchmark_metrics.json").write_text(
                    json.dumps({"episodes": [record]}), encoding="utf-8"
                )
            report = REPORT.aggregate_queue(root, expected_episodes=[1])

        self.assertFalse(report["complete"])
        self.assertEqual(report["methods"]["M1"]["valid_episodes"], 0)
        m1_row = next(row for row in report["rows"] if row["method"] == "M1")
        self.assertEqual(m1_row["status"], "failed")
        self.assertIn("runner_error", m1_row["failure_reasons"])
        self.assertEqual(report["paired_statistics"]["paired_episodes"], 0)
        self.assertTrue(any(item.get("reason") == "all_attempts_failed" for item in report["failures"]))

    def test_duplicate_valid_attempts_are_ambiguous_not_silently_picked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, override in enumerate((0.1, 0.2)):
                directory = root / f"m0_retry_{index}"
                directory.mkdir()
                (directory / "benchmark_metrics.json").write_text(
                    json.dumps({"episodes": [episode_record(1, success=True, spl=0.5, navigation_error=3.0, override_fraction=override)]}),
                    encoding="utf-8",
                )
            directory = root / "m1"
            directory.mkdir()
            (directory / "benchmark_metrics.json").write_text(
                json.dumps({"episodes": [episode_record(1, success=True, spl=0.6, navigation_error=2.0)]}),
                encoding="utf-8",
            )
            report = REPORT.aggregate_queue(root, expected_episodes=[1])

        m0_row = next(row for row in report["rows"] if row["method"] == "M0")
        self.assertEqual(m0_row["status"], "ambiguous_duplicate")
        self.assertEqual(report["paired_statistics"]["paired_episodes"], 0)

    def test_seed_and_strict_contract_are_required(self) -> None:
        record = episode_record(3, success=True, spl=0.4, navigation_error=2.0)
        record["episode_seed"] = 100
        record["strict_parity_contract"] = False
        row = REPORT.validate_candidate(
            REPORT.Candidate("fixture", "M0", "episode_3", record, {}, ())
        )
        self.assertEqual(row["status"], "failed")
        self.assertIn("episode_seed_mismatch", row["failure_reasons"])
        self.assertIn("strict_parity_contract_not_true", row["failure_reasons"])

    def test_exact_wilcoxon_and_bootstrap_are_repeatable(self) -> None:
        self.assertEqual(REPORT.wilcoxon_signed_rank_pvalue([1.0, 2.0, 3.0]), 0.25)
        self.assertIsNone(REPORT.wilcoxon_signed_rank_pvalue([0.0, 0.0]))
        first = REPORT.paired_bootstrap_ci([1.0, -1.0, 0.5], seed=7, samples=200)
        second = REPORT.paired_bootstrap_ci([1.0, -1.0, 0.5], seed=7, samples=200)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
