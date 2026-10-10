#!/usr/bin/env python3
"""Compare paired M1 and M0 outcomes across 20 preregistered DynaNav task families.

The 85 benchmark entries are not 85 independent routes. The primary estimand
gives equal weight to each fixed family; episode-weighted results remain visible
as descriptive context. This reads an existing matrix summary and runs no sim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subset_utils import load_yaml_mapping, sha256_file


BENCHMARK_SHA256 = "62d8abec03dc793dad9da351c5c5652bcd854014f29a91880309bd70c128b08c"
BOOTSTRAP_SEED = 20261003
BOOTSTRAP_SAMPLES = 10_000
FAMILY_SPECS = (
    *(("hospital", start, start + 4) for start in (1, 6, 11, 16, 21)),
    *(("office", start, start + 4) for start in (26, 31, 36, 41, 46)),
    *(("outdoor", start, start + 1) for start in (51, 53, 55, 57, 59)),
    *(("warehouse", start, start + 4) for start in (61, 66, 71, 76, 81)),
)
METRICS = ("success_rate", "spl", "navigation_error_m")


class InvalidMatrix(ValueError):
    """The source matrix cannot support a complete paired comparison."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidMatrix(message)


def _finite_number(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    number = float(value)
    _require(math.isfinite(number), f"{label} must be finite")
    return number


def _family_lookup(benchmark: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    keys = {key for key in benchmark if re.fullmatch(r"episode_\d+", str(key))}
    expected_keys = {f"episode_{index}" for index in range(1, 86)}
    _require(keys == expected_keys, "benchmark must contain exactly episode_1..episode_85")
    lookup: dict[int, dict[str, Any]] = {}
    families: list[dict[str, Any]] = []
    unique_targets: set[tuple[str, tuple[float, ...]]] = set()
    for environment, first, last in FAMILY_SPECS:
        indices = list(range(first, last + 1))
        configs = [benchmark[f"episode_{index}"] for index in indices]
        _require(all(isinstance(item, dict) for item in configs), f"family {first}-{last} has invalid config")
        scenes = {str(item.get("scene")) for item in configs}
        goals = {tuple(item.get("goal", ())) for item in configs}
        starts = {tuple(item.get("start", ())) for item in configs}
        _require(len(scenes) == len(goals) == 1, f"family {first}-{last} no longer has one scene/goal")
        _require(all(start and goal for start in starts for goal in goals), f"family {first}-{last} lacks start/goal")
        target = (next(iter(scenes)), next(iter(goals)))
        _require(target not in unique_targets, f"family {first}-{last} duplicates another scene/goal")
        unique_targets.add(target)
        family = {
            "family_id": f"{environment}_{first:02d}_{last:02d}",
            "environment": environment,
            "episodes": indices,
            "episode_count": len(indices),
            "distinct_start_count": len(starts),
        }
        families.append(family)
        for index in indices:
            _require(index not in lookup, f"episode_{index} belongs to two families")
            lookup[index] = family
    _require(set(lookup) == set(range(1, 86)) and len(families) == 20, "family registration does not cover 85 episodes")
    return lookup, families


def _validate_rows(summary: dict[str, Any], benchmark: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    _require(summary.get("complete") is True, "matrix summary is not complete")
    _require(summary.get("official_episode_count") == 85, "matrix summary is not the complete 85-episode set")
    _require(summary.get("missing_or_invalid") == [], "matrix summary contains missing/invalid runs")
    for field, expected in (("benchmark_episode_count", 85), ("selected_episode_indices", list(range(1, 86)))):
        if field in summary:
            _require(summary[field] == expected, f"matrix summary {field} does not match the official 85 episodes")
    rows = summary.get("rows")
    _require(isinstance(rows, list), "matrix summary rows must be an array")
    if "completed_result_rows" in summary:
        _require(summary["completed_result_rows"] == len(rows), "completed_result_rows differs from rows length")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, dict), "matrix summary contains a non-object row")
        method = row.get("method")
        if method not in ("M0", "M1"):
            continue
        index = row.get("episode")
        _require(isinstance(index, int) and not isinstance(index, bool) and 1 <= index <= 85, f"{method} has invalid episode id")
        key = (method, index)
        _require(key not in indexed, f"duplicate {method} episode_{index}")
        run_id = row.get("run_id")
        _require(isinstance(run_id, str) and re.search(rf"(?:^|_){method.lower()}_e{index:03d}$", run_id) is not None,
                 f"{method} episode_{index} has inconsistent run_id")
        _require(row.get("process_status") in ("passed", "cached"), f"{run_id} did not pass")
        _require(isinstance(row.get("success"), bool), f"{run_id} success must be boolean")
        spl = _finite_number(row.get("spl"), f"{run_id} spl")
        nav = _finite_number(row.get("navigation_error_m"), f"{run_id} navigation_error_m")
        _require(0.0 <= spl <= 1.0 and nav >= 0.0, f"{run_id} has out-of-range metrics")
        official_people = benchmark[f"episode_{index}"].get("num_people")
        _require(row.get("official_num_people") == official_people, f"{run_id} official people count differs from benchmark")
        _require(row.get("runtime_num_people") == 0, f"{run_id} is not the static-person compatibility run")
        indexed[key] = row
    expected = {(method, index) for method in ("M0", "M1") for index in range(1, 86)}
    _require(set(indexed) == expected, f"missing {len(expected - set(indexed))} M0/M1 paired rows")
    return indexed


def _percentile(sorted_values: list[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def family_bootstrap_ci(values: Sequence[float], *, seed: int = BOOTSTRAP_SEED, samples: int = BOOTSTRAP_SAMPLES) -> list[float]:
    _require(len(values) == 20, "bootstrap requires 20 family differences")
    _require(samples >= 2, "bootstrap requires at least two resamples")
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def exact_sign_permutation_p(values: Sequence[float]) -> float:
    """Two-sided paired sign randomization over all 2**20 family assignments."""
    _require(len(values) == 20, "sign permutation requires 20 family differences")
    differences = [float(value) for value in values if value != 0.0]
    _require(all(math.isfinite(value) for value in differences), "non-finite family difference")
    if not differences:
        return 1.0
    observed_sum = math.fsum(differences)
    sums = [0.0]
    for difference in differences:
        sums = [value + difference for value in sums] + [value - difference for value in sums]
    tolerance = 1e-12 * max(1.0, abs(observed_sum))
    extreme = sum(abs(value) + tolerance >= abs(observed_sum) for value in sums)
    return extreme / len(sums)


def build_report(summary: dict[str, Any], benchmark: dict[str, Any], *, summary_sha256: str,
                 benchmark_sha256: str, bootstrap_samples: int = BOOTSTRAP_SAMPLES) -> dict[str, Any]:
    _require(benchmark_sha256 == BENCHMARK_SHA256, "benchmark YAML hash differs from the preregistered official config")
    lookup, families = _family_lookup(benchmark)
    indexed = _validate_rows(summary, benchmark)
    episodes: list[dict[str, Any]] = []
    for index in range(1, 86):
        m0, m1 = indexed[("M0", index)], indexed[("M1", index)]
        m0_metrics = {"success_rate": float(m0["success"]), "spl": float(m0["spl"]),
                      "navigation_error_m": float(m0["navigation_error_m"])}
        m1_metrics = {"success_rate": float(m1["success"]), "spl": float(m1["spl"]),
                      "navigation_error_m": float(m1["navigation_error_m"])}
        episodes.append({
            "episode": index,
            "family_id": lookup[index]["family_id"],
            "official_num_people": m0["official_num_people"],
            "run_id_m0": m0["run_id"],
            "run_id_m1": m1["run_id"],
            "m0": m0_metrics,
            "m1": m1_metrics,
            "delta_m1_minus_m0": {metric: m1_metrics[metric] - m0_metrics[metric] for metric in METRICS},
        })

    for family in families:
        members = [episodes[index - 1] for index in family["episodes"]]
        family["m0_mean"] = {metric: statistics.fmean(item["m0"][metric] for item in members) for metric in METRICS}
        family["m1_mean"] = {metric: statistics.fmean(item["m1"][metric] for item in members) for metric in METRICS}
        family["delta_m1_minus_m0"] = {
            metric: statistics.fmean(item["delta_m1_minus_m0"][metric] for item in members) for metric in METRICS
        }

    metrics: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        differences = [family["delta_m1_minus_m0"][metric] for family in families]
        metrics[metric] = {
            "family_equal_mean_delta": statistics.fmean(differences),
            "family_bootstrap_ci_95": family_bootstrap_ci(differences, samples=bootstrap_samples),
            "family_sign_permutation_p_two_sided": exact_sign_permutation_p(differences),
            "episode_equal_mean_delta": statistics.fmean(item["delta_m1_minus_m0"][metric] for item in episodes),
            "m0_episode_mean": statistics.fmean(item["m0"][metric] for item in episodes),
            "m1_episode_mean": statistics.fmean(item["m1"][metric] for item in episodes),
        }
    return {
        "schema_version": 1,
        "comparison": "M1 minus M0",
        "source_summary_sha256": summary_sha256,
        "benchmark_sha256": benchmark_sha256,
        "episode_count": 85,
        "family_count": 20,
        "estimand": "equal-weight mean of 20 preregistered task-family paired differences",
        "inference": {"bootstrap_unit": "task_family", "bootstrap_samples": bootstrap_samples,
                      "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_interval": "percentile_95",
                      "sign_permutation": "exact_two_sided_all_family_sign_flips"},
        "scope": "Isaac Sim 6 static-person compatibility run; official_num_people is metadata, runtime_num_people=0",
        "limitations": [
            "Episode-level seed and initial-state equality cannot be independently reverified from summary rows.",
            "Some members of one task family have different start poses (notably episodes 57-58).",
            "This is not an official dynamic-person DynaNav benchmark score.",
        ],
        "legacy_m0_seed_metadata_missing_count": sum(bool(indexed[("M0", i)].get("seed_metadata_missing")) for i in range(1, 86)),
        "metrics": metrics,
        "families": families,
        "episodes": episodes,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="complete 85-episode matrix summary JSON")
    parser.add_argument("--benchmark", type=Path, default=Path(__file__).resolve().parents[2] / "benchmark_full.yaml")
    parser.add_argument("--output", type=Path, required=True, help="new JSON path; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        _require(isinstance(summary, dict), "summary must be a JSON object")
        benchmark = load_yaml_mapping(args.benchmark)
        report = build_report(summary, benchmark, summary_sha256=sha256_file(args.summary),
                              benchmark_sha256=sha256_file(args.benchmark))
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, json.JSONDecodeError, InvalidMatrix) as exc:
        parser.exit(2, f"route-family analysis failed: {exc}\n")
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
