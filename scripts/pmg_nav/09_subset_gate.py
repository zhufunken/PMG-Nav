#!/usr/bin/env python3
"""Gate a PMG subset result before allowing an expensive full benchmark run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from subset_utils import evaluate_subset_gate, load_manifest, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="subset manifest JSON")
    parser.add_argument("--results", type=Path, required=True, help="subset matrix summary JSON")
    parser.add_argument("--output", type=Path, required=True, help="gate result JSON")
    parser.add_argument(
        "--candidate-method",
        default="M1",
        help="method to compare against M0; core MVP defaults to M1",
    )
    parser.add_argument("--baseline-method", default="M0")
    parser.add_argument("--min-success-rate-delta", type=float, default=0.0)
    parser.add_argument("--min-spl-delta", type=float, default=0.0)
    parser.add_argument("--max-navigation-error-delta", type=float, default=0.0)
    parser.add_argument(
        "--allow-equal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="allow a metric equal to its threshold (default: true)",
    )
    parser.add_argument(
        "--require-improvement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require at least one positive primary-metric change (default: true)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    results_path = args.results.resolve()
    manifest = load_manifest(manifest_path)
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read result JSON {results_path}: {exc}") from exc
    if not isinstance(results, dict):
        raise SystemExit("result JSON must contain an object")
    gate = evaluate_subset_gate(
        manifest,
        results,
        candidate_method=args.candidate_method,
        baseline_method=args.baseline_method,
        require_improvement=args.require_improvement,
        allow_equal=args.allow_equal,
        min_success_rate_delta=args.min_success_rate_delta,
        min_spl_delta=args.min_spl_delta,
        max_navigation_error_delta=args.max_navigation_error_delta,
    )
    manifest_sha256 = sha256_file(manifest_path)
    gate["manifest"] = str(manifest_path)
    gate["manifest_sha256"] = manifest_sha256
    gate["results"] = str(results_path)
    # ``06_finalize_official_matrix.py`` names these fields
    # ``subset_manifest``/``subset_manifest_sha256``; accept the generic names
    # too so hand-produced result summaries remain unambiguous.
    result_manifest = results.get("manifest", results.get("subset_manifest"))
    result_manifest_sha256 = results.get("manifest_sha256", results.get("subset_manifest_sha256"))
    if result_manifest_sha256 != manifest_sha256:
        gate["passed"] = False
        gate.setdefault("reasons", []).append(
            "result summary manifest_sha256 does not match the requested subset manifest"
        )
    if not isinstance(result_manifest, str):
        gate["passed"] = False
        gate.setdefault("reasons", []).append(
            "result summary does not identify the requested subset manifest"
        )
    elif Path(result_manifest).resolve() != manifest_path:
        gate["passed"] = False
        gate.setdefault("reasons", []).append(
            "result summary manifest path does not match the requested subset manifest"
        )
    write_json(args.output, gate)
    print(json.dumps(gate, ensure_ascii=True, indent=2))
    if not gate["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
