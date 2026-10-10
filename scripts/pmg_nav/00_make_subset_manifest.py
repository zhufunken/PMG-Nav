#!/usr/bin/env python3
"""Create a deterministic representative subset manifest from benchmark YAML."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from subset_utils import DEFAULT_METHODS, DEFAULT_ROUTE_QUANTIZATION_M, load_yaml_mapping, make_manifest, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True, help="source benchmark_full.yaml")
    parser.add_argument("--output", type=Path, required=True, help="JSON subset manifest path")
    parser.add_argument(
        "--size",
        type=int,
        default=24,
        help="number of episodes to select (default: 24; covers every route family in benchmark_full.yaml)",
    )
    parser.add_argument("--seed", type=int, default=20260905, help="stable selection seed")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), help="methods the runner will pair")
    parser.add_argument(
        "--route-quantization-m",
        type=float,
        default=DEFAULT_ROUTE_QUANTIZATION_M,
        help="coordinate quantization used for route families (default: 10 m)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size <= 0:
        raise SystemExit("--size must be positive")
    source = load_yaml_mapping(args.benchmark)
    manifest = make_manifest(
        source,
        benchmark_path=args.benchmark,
        seed=args.seed,
        size=args.size,
        methods=args.methods,
        quantization_m=args.route_quantization_m,
    )
    write_json(args.output, manifest)
    print(
        f"wrote {args.output}: {manifest['selection']['selected_size']}/{manifest['benchmark_episode_count']} episodes "
        f"(seed={args.seed})"
    )


if __name__ == "__main__":
    main()
