#!/usr/bin/env python3
"""Estimate wall-clock time and AutoDL cost from measured aggregate throughput."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Workload:
    label: str
    environment_steps: int
    evidence: str


WORKLOADS = {
    "ovon_dagger": Workload(
        label="HM3D-OVON DAgger",
        environment_steps=150_000_000,
        evidence="official rnn_dagger.yaml total_num_steps",
    ),
    "goat_monolithic": Workload(
        label="GOAT-Bench monolithic PPO",
        environment_steps=270_000 * 32 * 64,
        evidence="official config: num_updates × num_environments × rollout steps",
    ),
}


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=sorted(WORKLOADS), required=True)
    parser.add_argument(
        "--instance-price",
        type=positive_float,
        required=True,
        help="Whole-instance price in CNY/hour.",
    )
    parser.add_argument(
        "--throughput",
        type=positive_float,
        required=True,
        help="Measured aggregate environment steps/second for the whole instance.",
    )
    parser.add_argument("--seeds", type=positive_int, default=1)
    parser.add_argument(
        "--overhead",
        type=positive_float,
        default=1.30,
        help="Multiplier for evaluation, restart, and checkpoint overhead (default: 1.30).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workload = WORKLOADS[args.baseline]
    single_hours = workload.environment_steps / args.throughput / 3600
    training_hours = single_hours * args.seeds
    budget_hours = training_hours * args.overhead
    training_cost = training_hours * args.instance_price
    budget_cost = budget_hours * args.instance_price

    print(f"baseline: {workload.label}")
    print(f"evidence: {workload.evidence}")
    print(f"environment_steps_per_seed: {workload.environment_steps:,}")
    print(f"aggregate_throughput_steps_per_second: {args.throughput:,.2f}")
    print(f"single_seed_hours: {single_hours:,.2f}")
    print(f"seeds: {args.seeds}")
    print(f"training_hours: {training_hours:,.2f}")
    print(f"training_cost_cny: {training_cost:,.2f}")
    print(f"budget_overhead_multiplier: {args.overhead:.2f}")
    print(f"recommended_budget_hours: {budget_hours:,.2f}")
    print(f"recommended_budget_cny: {budget_cost:,.2f}")


if __name__ == "__main__":
    main()
