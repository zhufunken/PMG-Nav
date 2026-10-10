#!/usr/bin/env python3
"""Focused tests for the preregistered 85-episode family analysis."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("analyze_official_route_families.py")
SPEC = importlib.util.spec_from_file_location("route_family_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
BENCHMARK = SCRIPT.parents[2] / "benchmark_full.yaml"


class RouteFamilyAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = MODULE.load_yaml_mapping(BENCHMARK)

    def setUp(self) -> None:
        rows = []
        for method in ("M0", "M1"):
            for index in range(1, 86):
                rows.append({
                    "run_id": f"test_{method.lower()}_e{index:03d}",
                    "method": method,
                    "episode": index,
                    "official_num_people": self.benchmark[f"episode_{index}"]["num_people"],
                    "runtime_num_people": 0,
                    "process_status": "passed",
                    "success": method == "M1",
                    "spl": 1.0 if method == "M1" else 0.0,
                    "navigation_error_m": 0.0 if method == "M1" else 1.0,
                })
        self.summary = {"complete": True, "official_episode_count": 85,
                        "missing_or_invalid": [], "completed_result_rows": 170, "rows": rows}

    def analyze(self) -> dict:
        return MODULE.build_report(self.summary, self.benchmark, summary_sha256="synthetic",
                                   benchmark_sha256=MODULE.BENCHMARK_SHA256, bootstrap_samples=100)

    def test_complete_pairing_and_deterministic_statistics(self) -> None:
        report = self.analyze()
        self.assertEqual((report["episode_count"], report["family_count"]), (85, 20))
        self.assertEqual(report["families"][13]["episodes"], [57, 58])
        self.assertEqual(report["families"][13]["distinct_start_count"], 2)
        for metric, expected in (("success_rate", 1.0), ("spl", 1.0), ("navigation_error_m", -1.0)):
            values = report["metrics"][metric]
            self.assertAlmostEqual(values["family_equal_mean_delta"], expected)
            self.assertEqual(values["family_bootstrap_ci_95"], [expected, expected])
            self.assertAlmostEqual(values["family_sign_permutation_p_two_sided"], 2 / 2**20)

    def test_episode_and_family_weighting_are_distinct(self) -> None:
        # Only a two-entry outdoor family improves, so it receives 1/20
        # inferential weight rather than 2/85 episode weight.
        for row in self.summary["rows"]:
            if row["method"] == "M1":
                row["success"] = row["episode"] in (51, 52)
            else:
                row["success"] = False
        report = self.analyze()
        result = report["metrics"]["success_rate"]
        self.assertAlmostEqual(result["family_equal_mean_delta"], 1 / 20)
        self.assertAlmostEqual(result["episode_equal_mean_delta"], 2 / 85)

    def test_incomplete_duplicate_and_invalid_metrics_are_rejected(self) -> None:
        for mutation in ("incomplete", "duplicate", "nan", "runtime_people", "missing"):
            with self.subTest(mutation=mutation):
                original = copy.deepcopy(self.summary)
                if mutation == "incomplete":
                    self.summary["complete"] = False
                elif mutation == "duplicate":
                    self.summary["rows"].append(copy.deepcopy(self.summary["rows"][0]))
                    self.summary["completed_result_rows"] += 1
                elif mutation == "nan":
                    self.summary["rows"][0]["navigation_error_m"] = float("nan")
                elif mutation == "runtime_people":
                    self.summary["rows"][0]["runtime_num_people"] = 10
                else:
                    self.summary["rows"].pop()
                    self.summary["completed_result_rows"] -= 1
                with self.assertRaises(MODULE.InvalidMatrix):
                    self.analyze()
                self.summary = original

    def test_wrong_benchmark_hash_is_rejected(self) -> None:
        with self.assertRaisesRegex(MODULE.InvalidMatrix, "hash differs"):
            MODULE.build_report(self.summary, self.benchmark, summary_sha256="synthetic",
                                benchmark_sha256="wrong", bootstrap_samples=100)

    def test_zero_differences_have_p_one(self) -> None:
        self.assertEqual(MODULE.exact_sign_permutation_p([0.0] * 20), 1.0)


if __name__ == "__main__":
    unittest.main()
