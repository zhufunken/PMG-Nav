#!/usr/bin/env python3
"""Run the staged no-dynamic-people Stage 2 family matrix.

DynaNav benchmark_full contains 85 entries, mostly the same 20 route families
repeated with increasing dynamic-person counts. The accepted Isaac Sim 6
compatibility runtime cannot run Replicator Agent 0.x dynamic people, so this
driver evaluates every unique route family once with `num_people=0`. It keeps
the original scene, start, goal, instruction, timeout, and robot type.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from subset_utils import dump_yaml_mapping, load_yaml_mapping


ROUTE_FAMILY_EPISODES = (1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 53, 55, 57, 59, 61, 66, 71, 76, 81)
METHODS = ("M0", "M1", "M2", "M3")
CORE_METHODS = ("M0", "M1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/ccfb-ticvla"))
    parser.add_argument(
        "--complexity",
        choices=("core", "full"),
        default="core",
        help="core runs only M0/M1; full is required before enabling M2/M3",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=None,
        help="override the profile methods; extra methods require --complexity full",
    )
    parser.add_argument("--episodes", nargs="+", type=int, default=list(ROUTE_FAMILY_EPISODES))
    parser.add_argument(
        "--m1-variant",
        choices=("strict", "simple_oracle"),
        default=None,
        help="M1 controller profile; simple_oracle is an explicitly labeled upper bound",
    )
    parser.add_argument("--graph-mode", choices=("full", "compact"), default=None)
    parser.add_argument("--max-graph-nodes", type=int, default=None)
    parser.add_argument("--max-edges-per-anchor", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def valid_result(path: Path, method: str, root: Path, run_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        episode = payload["episodes"][0]
        if episode.get("error") or payload.get("total_episodes") != 1:
            return False
        if method != "M0":
            pmg_path = root / "runtime/dynanav/pmg_metrics" / run_id / f"{episode['episode']}.json"
            if not pmg_path.is_file():
                return False
            pmg = json.loads(pmg_path.read_text(encoding="utf-8"))
            if pmg.get("observations", 0) < 2 or pmg.get("graph_nodes", 0) < 1:
                return False
        return True
    except (KeyError, IndexError, json.JSONDecodeError):
        return False


def main() -> None:
    args = parse_args()
    methods = tuple(args.methods or CORE_METHODS)
    if args.complexity == "core" and any(method not in CORE_METHODS for method in methods):
        raise SystemExit(
            "M2/M3 are disabled by the core MVP profile; pass --complexity full "
            "explicitly before enabling temporal/tools stages"
        )
    if args.complexity == "core" and args.m1_variant not in (None, "strict"):
        raise SystemExit(
            "simple_oracle is an upper-bound diagnostic; pass --complexity full "
            "before enabling it"
        )
    root = args.root.resolve()
    full_config = root / "src/TIC-VLA/DynaNav/configs/benchmark_full.yaml"
    source = load_yaml_mapping(full_config)
    config_root = root / "project/main/configs/pmg_nav/full"
    output_root = root / "outputs/stage2_full"
    log_root = root / "logs/stage2_full"
    default_prefix = "stage2_full_core" if args.complexity == "core" else "stage2_full"
    run_prefix = os.environ.get("STAGE2_RUN_PREFIX", default_prefix)
    # Core and full profiles must never share progress state: otherwise a
    # valid M0/M1 artifact from an older all-method run could silently satisfy
    # the new MVP invocation.
    status_path = output_root / ("progress.json" if run_prefix == "stage2_full" else f"{run_prefix}_progress.json")
    config_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    progress = {"started": datetime.now(timezone.utc).isoformat(), "runs": {}}
    if status_path.is_file():
        progress = json.loads(status_path.read_text(encoding="utf-8"))
    base_env = os.environ.copy()
    base_env.update(
        {
            "PMG_ROOT": str(root),
            "PROJECT_MAIN": str(root / "project/main"),
            "TICVLA_REPO_DIR": str(root / "src/TIC-VLA"),
            "TICVLA_BASE_MODEL_PATH": str(root / "models/InternVL3-1B"),
            "TICVLA_CHECKPOINT_PATH": str(root / "data/TIC-VLA/TIC-VLA-model.ckpt"),
            "TICVLA_DATA_ROOT": str(root / "data/TIC-VLA"),
            "TICVLA_OUTPUT_DIR": str(root / "outputs"),
            "ISAAC_SIM_ROOT": str(root / "isaacsim-6.0.1"),
            "TICVLA_ISAAC6_COMPAT": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "PMG_ENABLE_GOAL_ADAPTER": os.environ.get("PMG_ENABLE_GOAL_ADAPTER", "0"),
            "PMG_ENABLE_VECTOR_ADAPTER": os.environ.get("PMG_ENABLE_VECTOR_ADAPTER", "0"),
        }
    )
    # Do not let a previous exploratory shell leak optional adapters or the
    # simple-oracle upper bound into the core MVP profile.
    if args.complexity == "core":
        base_env["PMG_ENABLE_GOAL_ADAPTER"] = "0"
        base_env["PMG_ENABLE_VECTOR_ADAPTER"] = "0"
    requested_variant = args.m1_variant or (
        "strict" if args.complexity == "core" and "M1" in methods else None
    )
    requested_graph_mode = args.graph_mode or (
        "full" if args.complexity == "core" and "M1" in methods else None
    )
    if args.complexity == "full":
        requested_variant = requested_variant or os.environ.get("STAGE2_M1_VARIANT")
        requested_graph_mode = requested_graph_mode or os.environ.get("PMG_GRAPH_MODE")
    requested_control_policy = os.environ.get(
        "PMG_CONTROL_POLICY",
        "policy_only" if "M1" not in methods else (
            "terminal" if requested_variant == "simple_oracle" else "recovery"
        ),
    ).strip().lower()
    requested_max_nodes = args.max_graph_nodes
    if requested_max_nodes is None and os.environ.get("PMG_MAX_GRAPH_NODES"):
        requested_max_nodes = int(os.environ["PMG_MAX_GRAPH_NODES"])
    requested_max_edges = args.max_edges_per_anchor
    if requested_max_edges is None and os.environ.get("PMG_MAX_EDGES_PER_ANCHOR"):
        requested_max_edges = int(os.environ["PMG_MAX_EDGES_PER_ANCHOR"])
    if requested_max_nodes is not None and requested_max_nodes < 2:
        raise ValueError("--max-graph-nodes must be >= 2")
    if requested_max_edges is not None and requested_max_edges < 1:
        raise ValueError("--max-edges-per-anchor must be >= 1")
    if requested_variant and "M1" not in methods:
        raise ValueError("--m1-variant requires M1 in --methods")
    profile = {
        "complexity": args.complexity,
        "methods": list(methods),
        "m1_variant": requested_variant if "M1" in methods else None,
        "graph_mode": requested_graph_mode if "M1" in methods else None,
        "max_graph_nodes": requested_max_nodes if "M1" in methods else None,
        "max_edges_per_anchor": requested_max_edges if "M1" in methods else None,
        "control_policy": requested_control_policy,
        "terminal_enter_distance_m": float(os.environ.get("PMG_TERMINAL_ENTER_DISTANCE", "4.0")),
        "terminal_exit_distance_m": float(os.environ.get("PMG_TERMINAL_EXIT_DISTANCE", "5.0")),
        "run_prefix": None,
    }
    if requested_variant == "simple_oracle" and "simple_oracle" not in run_prefix:
        run_prefix += "_m1_simple_oracle"
    if requested_graph_mode == "compact" and "compact" not in run_prefix:
        run_prefix += "_compact"
    if requested_control_policy in {"terminal", "global", "policy_only"} and requested_control_policy not in run_prefix:
        run_prefix += f"_{requested_control_policy}"
    profile["run_prefix"] = run_prefix
    profile_path = output_root / f"{run_prefix}_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    for method in methods:
        for index in args.episodes:
            episode_key = f"episode_{index}"
            if episode_key not in source:
                raise KeyError(episode_key)
            run_id = f"{run_prefix}_{method.lower()}_e{index:03d}"
            result_path = root / "outputs/dynanav_closed_loop" / run_id / "benchmark_metrics.json"
            if not args.force and valid_result(result_path, method, root, run_id):
                progress["runs"][run_id] = {"status": "passed", "cached": True, "result": str(result_path)}
                status_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
                continue
            episode = dict(source[episode_key])
            episode["num_people"] = 0
            config = {"seed": int(source.get("seed", 666)), "navigation_method": "ticvla", "success_threshold": float(source.get("success_threshold", 1.5)), episode_key: episode}
            method_dir = config_root / method.lower()
            method_dir.mkdir(parents=True, exist_ok=True)
            config_path = method_dir / f"{episode_key}.yaml"
            config_path.write_text(dump_yaml_mapping(config), encoding="utf-8")
            env = dict(base_env)
            env["TICVLA_DYNANAV_RUN_ID"] = run_id
            env["PMG_METHOD"] = method
            # Keep the paired comparison fair: PMG is terminal-local by
            # default; unrestricted global control requires an explicit
            # ``PMG_CONTROL_POLICY=global`` upper-bound run.
            env["PMG_CONTROL_POLICY"] = (
                "policy_only"
                if method == "M0"
                else os.environ.get(
                    "PMG_CONTROL_POLICY",
                    "terminal" if method == "M1" and requested_variant == "simple_oracle" else "recovery",
                )
            )
            env["PMG_TERMINAL_ENTER_DISTANCE"] = os.environ.get("PMG_TERMINAL_ENTER_DISTANCE", "4.0")
            env["PMG_TERMINAL_EXIT_DISTANCE"] = os.environ.get("PMG_TERMINAL_EXIT_DISTANCE", "5.0")
            if method == "M1" and requested_variant:
                env["STAGE2_M1_VARIANT"] = requested_variant
                env["PMG_GRAPH_MODE"] = requested_graph_mode or (
                    "compact" if requested_variant == "simple_oracle" else "full"
                )
                env["PMG_MAX_GRAPH_NODES"] = str(requested_max_nodes or 8)
                env["PMG_MAX_EDGES_PER_ANCHOR"] = str(requested_max_edges or 4)
                env["PMG_SIMPLE_ORACLE_MIN_SCORE"] = os.environ.get("PMG_SIMPLE_ORACLE_MIN_SCORE", "0.0")
                env["PMG_SIMPLE_ORACLE_MIN_STANDOFF"] = os.environ.get("PMG_SIMPLE_ORACLE_MIN_STANDOFF", "0.75")
                env["PMG_SIMPLE_ORACLE_STOP_RADIUS"] = os.environ.get("PMG_SIMPLE_ORACLE_STOP_RADIUS", "0.60")
                env["PMG_SIMPLE_ORACLE_BLEND"] = os.environ.get("PMG_SIMPLE_ORACLE_BLEND", "1.0")
                env["PMG_TERMINAL_BLEND"] = os.environ.get("PMG_TERMINAL_BLEND", "1.0")
                env["PMG_TERMINAL_TRIGGER"] = os.environ.get("PMG_TERMINAL_TRIGGER", "handoff")
            log_path = log_root / f"{run_id}.log"
            command = ["bash", "scripts/08_run_dynanav_baseline.sh", str(config_path)]
            progress["runs"][run_id] = {"status": "running", "started": datetime.now(timezone.utc).isoformat(), "config": str(config_path)}
            status_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            started = time.time()
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=root / "project/main/autodl", env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            valid = valid_result(result_path, method, root, run_id)
            progress["runs"][run_id].update(
                {
                    "status": "passed" if completed.returncode == 0 and valid else "failed",
                    "exit_code": completed.returncode,
                    "finished": datetime.now(timezone.utc).isoformat(),
                    "wall_seconds": time.time() - started,
                    "result": str(result_path),
                    "log": str(log_path),
                }
            )
            status_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            print(json.dumps({"run_id": run_id, **progress["runs"][run_id]}, sort_keys=True), flush=True)
    progress["finished"] = datetime.now(timezone.utc).isoformat()
    progress["complexity"] = args.complexity
    progress["methods"] = list(methods)
    progress["passed"] = sum(item["status"] == "passed" for item in progress["runs"].values())
    progress["failed"] = sum(item["status"] == "failed" for item in progress["runs"].values())
    status_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    if progress["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
