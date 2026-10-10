#!/usr/bin/env python3
"""Unit tests for development smoke workload guards in the matrix runner."""

from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch


_SCRIPT = Path(__file__).with_name("05_run_official_matrix.py")
_SPEC = importlib.util.spec_from_file_location("official_matrix", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
validate_smoke_constraints = _MODULE.validate_smoke_constraints
disk_budget = _MODULE.disk_budget
run_groups = _MODULE.run_groups

_FINALIZER = Path(__file__).with_name("06_finalize_official_matrix.py")
_FINALIZER_SPEC = importlib.util.spec_from_file_location("official_finalizer", _FINALIZER)
assert _FINALIZER_SPEC is not None and _FINALIZER_SPEC.loader is not None
_FINALIZER_MODULE = importlib.util.module_from_spec(_FINALIZER_SPEC)
_FINALIZER_SPEC.loader.exec_module(_FINALIZER_MODULE)
wilcoxon_signed_rank_pvalue = getattr(_FINALIZER_MODULE, "_wilcoxon_signed_rank_pvalue", None)


class SmokeWorkloadContractTests(unittest.TestCase):
    def test_paired_order_keeps_each_m0_m1_pair_adjacent(self) -> None:
        self.assertEqual(
            run_groups(("M0", "M1"), [1, 2], paired_episode_order=True),
            [[("M0", 1), ("M1", 1)], [("M0", 2), ("M1", 2)]],
        )
        self.assertEqual(
            run_groups(("M0", "M1"), [1, 2], paired_episode_order=False),
            [[("M0", 1), ("M0", 2)], [("M1", 1), ("M1", 2)]],
        )

    def test_none_keeps_formal_workload(self) -> None:
        self.assertIsNone(
            validate_smoke_constraints(None, strict_parity=False, is_full=False)
        )

    def test_positive_cap_is_accepted_for_subset(self) -> None:
        self.assertEqual(
            validate_smoke_constraints(120, strict_parity=False, is_full=False), 120
        )

    def test_zero_and_negative_caps_are_rejected(self) -> None:
        for value in (0, -1):
            with self.assertRaises(SystemExit):
                validate_smoke_constraints(value, strict_parity=False, is_full=False)

    def test_smoke_cannot_claim_strict_parity_or_full_run(self) -> None:
        with self.assertRaises(SystemExit):
            validate_smoke_constraints(120, strict_parity=True, is_full=False)
        with self.assertRaises(SystemExit):
            validate_smoke_constraints(120, strict_parity=False, is_full=True)

    def test_disk_budget_accepts_sufficient_space(self) -> None:
        usage = type("Usage", (), {"free": 80 * 1024**3, "total": 500 * 1024**3})()
        with patch.object(_MODULE.shutil, "disk_usage", return_value=usage):
            state = disk_budget(Path("/tmp"), 60.0)
        self.assertAlmostEqual(state["free_gib"], 80.0)

    def test_disk_budget_rejects_low_space(self) -> None:
        usage = type("Usage", (), {"free": 59 * 1024**3, "total": 500 * 1024**3})()
        with patch.object(_MODULE.shutil, "disk_usage", return_value=usage):
            with self.assertRaises(SystemExit):
                disk_budget(Path("/tmp"), 60.0)


@unittest.skipIf(wilcoxon_signed_rank_pvalue is None, "finalizer lacks Wilcoxon helper")
class PairedStatisticsTests(unittest.TestCase):
    def test_wilcoxon_matches_exact_small_sample(self) -> None:
        # For three positive, untied differences, the two-sided exact p-value
        # is 2 / 2**3 = 0.25.
        self.assertAlmostEqual(wilcoxon_signed_rank_pvalue([1.0, 2.0, 3.0]), 0.25)

    def test_wilcoxon_discards_zero_differences(self) -> None:
        self.assertAlmostEqual(
            wilcoxon_signed_rank_pvalue([0.0, 1.0, -1.0]), 1.0
        )

    def test_wilcoxon_returns_none_without_nonzero_pairs(self) -> None:
        self.assertIsNone(wilcoxon_signed_rank_pvalue([0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
