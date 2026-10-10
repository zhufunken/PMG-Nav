#!/usr/bin/env python3
"""Run a manifest-selected DynaNav benchmark workload for Stage 2.

The default workload is the core M0/M1 comparison; M2/M3 are opt-in follow-up
methods.  A subset manifest selects a deterministic representative workload,
while a full 85-episode workload requires a passed subset gate.

The released ``benchmark_full.yaml`` is the only source of episode metadata.
Isaac Sim 6 compatibility cannot instantiate the released Replicator Agent 0.x
dynamic-people API, so the official count is preserved as metadata while the
runtime config explicitly records ``num_people=0``.  This distinction is part
of every result and is never hidden by the aggregation step.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
# Isaac Sim's Python launcher can reset the caller's working-directory path.
# Keep the sibling helper import explicit; this is also what the shell
# launcher uses when invoking the matrix script from a different directory.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from subset_utils import dump_yaml_mapping, load_manifest, load_yaml_mapping, manifest_episode_indices, sha256_file


METHODS = ("M0", "M1", "M2", "M3")
CORE_METHODS = ("M0", "M1")
OFFICIAL_EPISODES = tuple(range(1, 86))


def _env_flag(*names: str) -> bool:
    """Return true when any named environment switch is explicitly enabled."""
    truthy = {"1", "true", "yes", "on"}
    return any(str(os.environ.get(name, "")).strip().lower() in truthy for name in names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/ccfb-ticvla"))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=None,
        help="methods to pair; defaults to manifest methods or the core M0/M1 comparison",
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        default=None,
        help="explicit episode ids; ignored only when --subset-manifest is supplied with the same ids",
    )
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        default=None,
        help="deterministic subset manifest produced by 00_make_subset_manifest.py",
    )
    parser.add_argument(
        "--full-after-gate",
        type=Path,
        default=None,
        metavar="GATE_JSON",
        help="permit all 85 episodes only when this subset gate artifact has passed=true",
    )
    parser.add_argument(
        "--allow-full",
        action="store_true",
        help="explicit emergency override for a full 85-episode run (recorded in manifest)",
    )
    parser.add_argument(
        "--m1-variant",
        choices=("strict", "simple_oracle"),
        default=None,
        help="M1 controller profile; simple_oracle is an explicitly labeled upper bound",
    )
    parser.add_argument(
        "--graph-mode",
        choices=("full", "compact"),
        default=None,
        help="PMG graph retention policy for M1 simple_oracle",
    )
    parser.add_argument("--max-graph-nodes", type=int, default=None)
    parser.add_argument("--max-edges-per-anchor", type=int, default=None)
    parser.add_argument(
        "--strict-parity",
        action="store_true",
        help=(
            "enable fixed-horizon deterministic/greedy M0-M3 replay conditions; "
            "also available via STAGE2_STRICT_PARITY=1"
        ),
    )
    parser.add_argument(
        "--smoke-frames",
        type=int,
        default=None,
        metavar="N",
        help=(
            "development-only frame cap per episode; incompatible with "
            "strict parity and the full 85-episode workload"
        ),
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=float(os.environ.get("STAGE2_MIN_FREE_GIB", "60")),
        metavar="GIB",
        help="stop before a run when the filesystem containing --root has less free space (default: 60)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--paired-episode-order",
        action="store_true",
        help="run M0 then M1 for each episode before advancing to the next episode",
    )
    return parser.parse_args()


def validate_smoke_constraints(
    smoke_frames: int | None,
    *,
    strict_parity: bool,
    is_full: bool,
) -> int | None:
    """Validate the development-only smoke workload contract.

    A smoke run intentionally changes the episode horizon, so it cannot be
    reported as a strict paired result or unlock the 85-episode workload.
    Keep this check independent of filesystem state so callers can test it
    before any manifest, config, or artifact is written.
    """
    if smoke_frames is None:
        return None
    if smoke_frames <= 0:
        raise SystemExit("--smoke-frames must be a positive integer")
    if strict_parity:
        raise SystemExit("--smoke-frames cannot be combined with --strict-parity")
    if is_full:
        raise SystemExit(
            "--smoke-frames cannot be used with the full 85-episode workload"
        )
    return int(smoke_frames)


def disk_budget(root: Path, minimum_free_gib: float = 60.0) -> dict[str, float]:
    """Fail before launching Isaac when the experiment filesystem is too full.

    A full episode can leave several hundred MB of telemetry and renderer
    cache.  Checking the actual filesystem containing the run root avoids
    spending GPU time on a run that cannot archive its result.  The returned
    values are persisted in the workload manifest for auditability.
    """
    if minimum_free_gib < 0.0:
        raise ValueError("minimum_free_gib must be non-negative")
    usage = shutil.disk_usage(root)
    free_gib = float(usage.free) / (1024.0 ** 3)
    total_gib = float(usage.total) / (1024.0 ** 3)
    if free_gib < minimum_free_gib:
        raise SystemExit(
            f"insufficient free space on {root}: {free_gib:.2f} GiB available, "
            f"minimum is {minimum_free_gib:.2f} GiB"
        )
    return {"free_gib": free_gib, "total_gib": total_gib, "minimum_free_gib": float(minimum_free_gib)}


def _result_path(root: Path, run_id: str) -> Path:
    return root / "outputs/dynanav_closed_loop" / run_id / "benchmark_metrics.json"


def run_groups(methods: tuple[str, ...], episode_indices: list[int], paired_episode_order: bool) -> list[list[tuple[str, int]]]:
    if paired_episode_order:
        return [[(method, index) for method in methods] for index in episode_indices]
    return [[(method, index) for index in episode_indices] for method in methods]


def _artifacts_complete(root: Path, run_id: str) -> bool:
    """Reject simulator crashes that leave a partial benchmark JSON behind."""
    path = root / "outputs/dynanav_closed_loop" / run_id / "telemetry/artifacts.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(payload.get("rgb_frames", 0) or 0) > 0 and int(payload.get("trajectory_points", 0) or 0) > 0


def valid_result(root: Path, run_id: str, method: str, episode_key: str, official_people: int) -> bool:
    path = _result_path(root, run_id)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        episode = payload["episodes"][0]
        if episode.get("episode") != episode_key or episode.get("error"):
            return False
        episode_index = int(episode_key.rsplit("_", 1)[-1])
        expected_seed = 666 + episode_index
        if int(episode.get("episode_seed", -1)) != expected_seed:
            return False
        if int(episode.get("official_num_people", -1)) != official_people:
            return False
        if int(episode.get("runtime_num_people", -1)) != 0:
            return False
        if not isinstance(episode.get("success"), bool):
            return False
        navigation_error = float(episode["navigation_error"])
        spl = float(episode["spl"])
        duration = float(episode["duration"])
        if not all(math.isfinite(value) for value in (navigation_error, spl, duration)):
            return False
        if navigation_error < 0.0 or not 0.0 <= spl <= 1.0 or duration < 0.0:
            return False
        if not _artifacts_complete(root, run_id):
            return False
        if method != "M0":
            pmg_path = root / "runtime/dynanav/pmg_metrics" / run_id / f"{episode_key}.json"
            graph_path = pmg_path.with_suffix(".graph.json")
            if not pmg_path.is_file() or not graph_path.is_file():
                return False
            pmg = json.loads(pmg_path.read_text(encoding="utf-8"))
            if (
                pmg.get("method") != method
                or int(pmg.get("observations", 0) or 0) < 2
                or int(pmg.get("graph_nodes", 0) or 0) < 1
            ):
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _load_gate(path: Path, subset_manifest: Path | None) -> dict:
    """Require a passed, machine-readable gate tied to the selected subset."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read full-run gate {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("passed") is not True:
        raise SystemExit(f"full-run gate must contain passed=true: {path}")
    if subset_manifest is not None:
        expected_hash = sha256_file(subset_manifest)
        recorded_hash = payload.get("manifest_sha256")
        recorded_path = payload.get("manifest")
        path_matches = False
        if isinstance(recorded_path, str):
            path_matches = Path(recorded_path).resolve() == subset_manifest.resolve()
        if recorded_hash != expected_hash or not path_matches:
            raise SystemExit(
                "full-run gate does not match --subset-manifest; "
                "regenerate the gate from the exact manifest"
            )
    else:
        # A full run deliberately omits --subset-manifest because that flag
        # selects the subset.  Require the gate to carry a concrete manifest
        # path and digest so an unrelated passed JSON cannot unlock 85 eps.
        recorded_path = payload.get("manifest")
        recorded_hash = payload.get("manifest_sha256")
        if not isinstance(recorded_path, str) or not recorded_hash:
            raise SystemExit(
                "full-run gate must identify its subset manifest and manifest_sha256"
            )
        manifest_path = Path(recorded_path).resolve()
        if not manifest_path.is_file() or sha256_file(manifest_path) != recorded_hash:
            raise SystemExit("full-run gate references a missing or modified subset manifest")
    return payload


def _episode_selection(args: argparse.Namespace, source: dict, subset_manifest: Path | None) -> tuple[list[int], dict]:
    """Resolve episode ids and enforce the full-run gate before side effects."""

    explicit = list(args.episodes) if args.episodes is not None else None
    manifest_payload: dict = {}
    if subset_manifest is not None:
        subset_manifest = subset_manifest.resolve()
        manifest_payload = load_manifest(subset_manifest)
        manifest_indices = manifest_episode_indices(manifest_payload)
        if explicit is not None and explicit != manifest_indices:
            raise SystemExit(
                "--episodes conflicts with --subset-manifest; omit --episodes or pass the manifest ids exactly"
            )
        indices = manifest_indices
    elif explicit is not None:
        indices = explicit
    else:
        indices = list(OFFICIAL_EPISODES)

    if not indices:
        raise SystemExit("episode selection is empty")
    if len(set(indices)) != len(indices):
        raise SystemExit("episode selection contains duplicate ids")
    missing = [index for index in indices if f"episode_{index}" not in source]
    if missing:
        raise SystemExit(f"benchmark episodes missing: {missing}")
    if any(index < 1 for index in indices):
        raise SystemExit("episode ids must be positive")

    is_full = set(indices) == set(OFFICIAL_EPISODES) and len(indices) == len(OFFICIAL_EPISODES)
    gate_payload: dict | None = None
    if is_full:
        if args.allow_full:
            gate_policy = "allow_full_override"
        elif args.full_after_gate is not None:
            gate_payload = _load_gate(args.full_after_gate.resolve(), subset_manifest)
            gate_policy = "passed_subset_gate"
        else:
            raise SystemExit(
                "full 85-episode run blocked: pass --full-after-gate GATE_JSON or explicit --allow-full"
            )
    else:
        if args.full_after_gate is not None:
            raise SystemExit("--full-after-gate is valid only when selecting all 85 official episodes")
        gate_policy = "subset"
    selection = {
        "is_full": is_full,
        "gate_policy": gate_policy,
        "episode_indices": indices,
        "episode_keys": [f"episode_{index}" for index in indices],
        "subset_manifest": str(subset_manifest) if subset_manifest is not None else None,
        "subset_manifest_sha256": sha256_file(subset_manifest) if subset_manifest is not None else None,
        "subset_manifest_selection": manifest_payload.get("selection") if manifest_payload else None,
        "gate_artifact": str(args.full_after_gate.resolve()) if gate_payload is not None else None,
    }
    return indices, selection


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    disk_state = disk_budget(root, args.min_free_gib)
    project_main = Path(
        os.environ.get("PROJECT_MAIN", str(SCRIPT_DIR.parents[1]))
    ).resolve()
    full_config = root / "src/TIC-VLA/DynaNav/configs/benchmark_full.yaml"
    source = load_yaml_mapping(full_config)
    subset_manifest = args.subset_manifest.resolve() if args.subset_manifest is not None else None
    if subset_manifest is not None:
        manifest_source = load_manifest(subset_manifest)
        expected_hash = manifest_source.get("benchmark_sha256")
        if expected_hash and expected_hash != sha256_file(full_config):
            raise SystemExit(
                "subset manifest benchmark_sha256 does not match the runner benchmark_full.yaml"
            )
    selected_methods = tuple(
        args.methods
        or (load_manifest(subset_manifest).get("methods") if subset_manifest is not None else CORE_METHODS)
    )
    if (
        not selected_methods
        or len(set(selected_methods)) != len(selected_methods)
        or any(method not in METHODS for method in selected_methods)
    ):
        raise SystemExit("methods must be a non-empty subset of M0/M1/M2/M3 without duplicates")
    episode_indices, selection = _episode_selection(args, source, subset_manifest)
    official_episode_keys = [f"episode_{index}" for index in episode_indices]

    config_root = root / "project/main/configs/pmg_nav/official_compat"
    output_root = root / "outputs/stage2_official_compat"
    log_root = root / "logs/stage2_official_compat"
    config_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    # Strict parity is deliberately opt-in.  When enabled, the same switches
    # are propagated to every method shard (including M0) so a later M0/M3
    # replay cannot accidentally compare a stochastic legacy baseline with a
    # deterministic candidate.  The resolved value is persisted in the
    # workload manifest for auditability.
    strict_parity = bool(args.strict_parity or _env_flag("STAGE2_STRICT_PARITY"))
    smoke_frames = validate_smoke_constraints(
        args.smoke_frames,
        strict_parity=strict_parity,
        is_full=bool(selection["is_full"]),
    )

    manifest = {
        "benchmark_config": str(full_config),
        "official_episode_count": len(official_episode_keys),
        "official_episode_keys": official_episode_keys,
        "methods": list(selected_methods),
        "intended_processes": len(official_episode_keys) * len(selected_methods),
        "runtime": "isaac_sim_6.0.1_compatibility",
        "pairing_seed_policy": "benchmark_seed_plus_episode_index; method-independent",
        "dynamic_people_note": "official_num_people is retained per episode; runtime_num_people is 0 because Replicator Agent 0.x is unavailable in Isaac Sim 6",
        "created": datetime.now(timezone.utc).isoformat(),
        "selection": selection,
        "strict_parity": strict_parity,
        "parity_fixed_horizon": strict_parity,
        "parity_deterministic": strict_parity,
        "smoke_mode": smoke_frames is not None,
        "smoke_frames": smoke_frames,
        "result_class": "development_smoke" if smoke_frames is not None else "formal",
        "disk_budget": disk_state,
        "paired_episode_order": args.paired_episode_order,
    }
    diagnostic_matrix = os.environ.get("STAGE2_DIAGNOSTIC_MATRIX", "0") == "1"
    if diagnostic_matrix:
        manifest["diagnostic_contract"] = {
            "M0": {"perception_source": "none", "action_enabled": False},
            "M1": {"perception_source": "oracle", "action_enabled": True},
            "M2": {"perception_source": "predicted", "action_enabled": True},
            "M3": {"perception_source": "oracle", "action_enabled": False},
        }

    # Independent method shards can run concurrently on one host.  Each shard
    # keeps its own atomic progress file while result artifacts remain keyed by
    # run_id, so the episode data never collide.
    status_path = Path(os.environ.get("STAGE2_PROGRESS_PATH", str(output_root / "progress.json")))
    status_path.parent.mkdir(parents=True, exist_ok=True)
    progress = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {
        "started": datetime.now(timezone.utc).isoformat(),
        "runs": {},
    }
    base_env = os.environ.copy()
    # The matrix CLI is the source of truth for horizon mode.  Do not let a
    # stale shell-level smoke variable silently shorten a formal run.
    base_env.pop("TICVLA_SMOKE_FRAMES", None)
    base_env.update(
        {
            "PMG_ROOT": str(root),
            "TICVLA_WORK_ROOT": str(root),
            # The run root contains configs/artifacts, while Isaac-6 helper
            # scripts (notably the navigation shim and artifact archiver) may
            # live in the canonical project tree.  Honor an explicit caller
            # path so a fresh run cannot silently lose those helpers.
            "PROJECT_MAIN": os.environ.get("PROJECT_MAIN", str(project_main)),
            "TICVLA_REPO_DIR": str(root / "src/TIC-VLA"),
            "TICVLA_BASE_MODEL_PATH": str(root / "models/InternVL3-1B"),
            "TICVLA_CHECKPOINT_PATH": str(root / "data/TIC-VLA/TIC-VLA-model.ckpt"),
            "TICVLA_DATA_ROOT": str(root / "data/TIC-VLA"),
            "TICVLA_OUTPUT_DIR": str(root / "outputs"),
            "ISAAC_SIM_ROOT": str(root / "isaacsim-6.0.1"),
            "TICVLA_ISAAC6_COMPAT": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "TICVLA_OFFICIAL_BENCHMARK": str(full_config),
            "PMG_ENABLE_GOAL_ADAPTER": os.environ.get("PMG_ENABLE_GOAL_ADAPTER", "0"),
            "PMG_ENABLE_VECTOR_ADAPTER": os.environ.get("PMG_ENABLE_VECTOR_ADAPTER", "0"),
        }
    )
    # Closed-loop diagnostics retain RGB files for the policy, but strict
    # parity gates do not need a second MP4 archive. Keep this opt-in so
    # legacy/full reporting runs preserve their historical artifacts.
    if strict_parity or _env_flag("STAGE2_SKIP_VIDEO", "PMG_SKIP_VIDEO"):
        base_env["PMG_SKIP_VIDEO"] = "1"
    if strict_parity:
        # Set both aliases to make the condition explicit in shell logs and
        # compatible with older runner snapshots.  The generated behavior
        # wrapper also honors the *_ACTIVE marker emitted by the runner.
        base_env.update(
            {
                "PMG_PARITY_FIXED_HORIZON": "1",
                "STAGE2_PARITY_FIXED_HORIZON": "1",
                "PMG_PARITY_DETERMINISTIC": "1",
                "STAGE2_PARITY_DETERMINISTIC": "1",
                "PMG_PARITY_MIN_ACTION_FRAME": os.environ.get(
                    "PMG_PARITY_MIN_ACTION_FRAME", "7"
                ),
                "PMG_TRACE_CONTENT_HASH": os.environ.get(
                    "PMG_TRACE_CONTENT_HASH", "1"
                ),
                # The legacy stuck/backup heuristic is callback-timing
                # sensitive and can desynchronize paired M0/M3 horizons.
                "STAGE2_DISABLE_STUCK_RECOVERY": "1",
            }
        )

    requested_variant = args.m1_variant or os.environ.get("STAGE2_M1_VARIANT")
    requested_graph_mode = args.graph_mode or os.environ.get("PMG_GRAPH_MODE")
    requested_control_policy = os.environ.get(
        "PMG_CONTROL_POLICY",
        "policy_only" if "M1" not in selected_methods else (
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
    if requested_variant and "M1" not in selected_methods:
        raise ValueError("--m1-variant requires M1 in --methods")
    run_prefix = os.environ.get("STAGE2_RUN_PREFIX")
    if not run_prefix:
        # Keep subset artifacts isolated from the gated 85-episode run.  An
        # explicit environment prefix remains available for retries or
        # independent experiments, but the default must match the selection.
        run_prefix = "stage2_official" if selection["is_full"] else "stage2_subset_core"
        if requested_variant == "simple_oracle":
            run_prefix += "_m1_simple_oracle"
        if requested_graph_mode == "compact":
            run_prefix += "_compact"
        if requested_control_policy in {"terminal", "global", "policy_only"}:
            run_prefix += f"_{requested_control_policy}"
        if strict_parity:
            # Never reuse a legacy stochastic artifact under a strict replay
            # workload.  The prefix is part of the cache key as well as the
            # manifest, so callers need not remember to pass --force.
            run_prefix += "_strict_parity"
        if smoke_frames is not None:
            run_prefix += f"_smoke{smoke_frames}"
    elif strict_parity and not run_prefix.endswith("_strict_parity"):
        # The same isolation applies when the caller supplies an explicit
        # prefix that would otherwise collide with a legacy run.
        run_prefix += "_strict_parity"
    elif smoke_frames is not None and f"_smoke{smoke_frames}" not in run_prefix:
        # Make caller-supplied smoke prefixes distinct from formal artifacts.
        run_prefix += f"_smoke{smoke_frames}"
    if smoke_frames is not None:
        # A development rerun must never overwrite an earlier smoke result,
        # even when the caller reuses the same explicit prefix.
        smoke_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_prefix += f"_{smoke_stamp}"
        config_root = config_root / run_prefix
        config_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"official_workload_manifest_{run_prefix}.json"
    manifest["m1_variant"] = requested_variant if "M1" in selected_methods else None
    manifest["graph_mode"] = requested_graph_mode if "M1" in selected_methods else None
    manifest["max_graph_nodes"] = requested_max_nodes if "M1" in selected_methods else None
    manifest["max_edges_per_anchor"] = requested_max_edges if "M1" in selected_methods else None
    manifest["control_policy"] = requested_control_policy
    manifest["terminal_enter_distance_m"] = float(os.environ.get("PMG_TERMINAL_ENTER_DISTANCE", "4.0"))
    manifest["terminal_exit_distance_m"] = float(os.environ.get("PMG_TERMINAL_EXIT_DISTANCE", "5.0"))
    manifest["simple_oracle_require_stable"] = os.environ.get(
        "PMG_SIMPLE_ORACLE_REQUIRE_STABLE", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    manifest["simple_oracle_stability_min_score"] = float(
        os.environ.get("PMG_SIMPLE_ORACLE_STABILITY_MIN_SCORE", "0.80")
    )
    manifest["paired_action_prefix_enabled"] = bool(
        strict_parity and "M1" in selected_methods
    )
    manifest["paired_action_prefix_policy"] = (
        "M1 replays paired M0 commands until PMG terminal hand-off"
        if manifest["paired_action_prefix_enabled"] else None
    )
    manifest["run_prefix"] = run_prefix
    manifest["smoke_mode"] = smoke_frames is not None
    manifest["smoke_frames"] = smoke_frames
    manifest["result_class"] = "development_smoke" if smoke_frames is not None else "formal"
    manifest["disk_budget"] = disk_state
    # A retry must be able to keep a fresh method prefix while replaying the
    # already locked M0 action trace.  Without an explicit source prefix,
    # strict-parity retries either collide with an existing artifact or look
    # for a non-existent M0 trace under the new retry prefix.
    paired_m0_prefix = os.environ.get("STAGE2_M0_RUN_PREFIX", run_prefix)
    manifest["paired_m0_run_prefix"] = paired_m0_prefix if strict_parity and "M1" in selected_methods else None
    manifest["strict_parity"] = strict_parity
    manifest["parity_fixed_horizon"] = strict_parity
    manifest["parity_deterministic"] = strict_parity
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for group in run_groups(selected_methods, episode_indices, args.paired_episode_order):
        for method, index in group:
            episode_key = f"episode_{index}"
            source_episode = dict(source[episode_key])
            official_people = int(source_episode.get("num_people", 0))
            run_id = f"{run_prefix}_{method.lower()}_e{index:03d}"
            result_path = _result_path(root, run_id)
            previous = progress.get("runs", {}).get(run_id, {})
            # A non-zero process is never a cache hit, even if it left a
            # superficially parseable result JSON behind.
            cacheable = previous.get("status") in {"passed", "cached"}
            if not args.force and cacheable and valid_result(root, run_id, method, episode_key, official_people):
                progress["runs"][run_id] = {
                    "status": "passed",
                    "cached": True,
                    "result_class": "development_smoke" if smoke_frames is not None else "formal",
                    "smoke_frames": smoke_frames,
                    "official_num_people": official_people,
                    "runtime_num_people": 0,
                    "result": str(result_path),
                }
                status_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
                continue

            episode = dict(source_episode)
            episode["official_num_people"] = official_people
            episode["num_people"] = 0
            config = {
                "seed": int(source.get("seed", 666)),
                "navigation_method": "ticvla",
                "success_threshold": float(source.get("success_threshold", 1.5)),
                episode_key: episode,
            }
            config_path = config_root / method.lower() / f"{episode_key}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(dump_yaml_mapping(config), encoding="utf-8")

            env = dict(base_env)
            env["TICVLA_DYNANAV_RUN_ID"] = run_id
            env["PMG_METHOD"] = method
            if smoke_frames is not None:
                # The Isaac-6 compatibility runner consumes this existing
                # switch and caps only the effective episode horizon.  It is
                # deliberately absent from formal runs.
                env["TICVLA_SMOKE_FRAMES"] = str(smoke_frames)
            env["STAGE2_STRICT_PARITY_ACTIVE"] = "1" if strict_parity else "0"
            # Fair M1 default: TIC-VLA owns the global route; PMG only gets a
            # local terminal hand-off.  ``global`` remains an explicit oracle
            # upper-bound switch and is recorded in each run manifest.
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
                env["PMG_SIMPLE_ORACLE_MIN_SCORE"] = os.environ.get("PMG_SIMPLE_ORACLE_MIN_SCORE", "0.0")
                env["PMG_SIMPLE_ORACLE_MIN_STANDOFF"] = os.environ.get("PMG_SIMPLE_ORACLE_MIN_STANDOFF", "0.75")
                env["PMG_SIMPLE_ORACLE_STOP_RADIUS"] = os.environ.get("PMG_SIMPLE_ORACLE_STOP_RADIUS", "0.60")
                env["PMG_SIMPLE_ORACLE_REQUIRE_STABLE"] = os.environ.get("PMG_SIMPLE_ORACLE_REQUIRE_STABLE", "1")
                env["PMG_SIMPLE_ORACLE_STABILITY_MIN_SCORE"] = os.environ.get("PMG_SIMPLE_ORACLE_STABILITY_MIN_SCORE", "0.80")
                env["PMG_SIMPLE_ORACLE_BLEND"] = os.environ.get("PMG_SIMPLE_ORACLE_BLEND", "1.0")
                env["PMG_TERMINAL_BLEND"] = os.environ.get("PMG_TERMINAL_BLEND", "1.0")
                env["PMG_TERMINAL_TRIGGER"] = os.environ.get("PMG_TERMINAL_TRIGGER", "handoff")
                env["PMG_GRAPH_MODE"] = requested_graph_mode or ("compact" if requested_variant == "simple_oracle" else "full")
                env["PMG_MAX_GRAPH_NODES"] = str(requested_max_nodes or 8)
                env["PMG_MAX_EDGES_PER_ANCHOR"] = str(requested_max_edges or 4)
                # Strict paired M1 uses the already archived M0 command trace
                # as its pre-handoff counterfactual.  This is deliberately
                # separate from the PMG controller and is recorded in the
                # manifest/trace; a missing prefix must fail closed in the
                # generated behavior instead of silently reverting to an
                # independently scheduled TIC-VLA trajectory.
                if strict_parity:
                    m0_run_id = f"{paired_m0_prefix}_m0_e{index:03d}"
                    m0_trace = _result_path(root, m0_run_id).parent / "action_trace.jsonl"
                    env["STAGE2_PAIRED_ACTION_PREFIX"] = "1"
                    env["PMG_POLICY_PREFIX_TRACE"] = str(m0_trace)
                    manifest.setdefault("paired_action_prefix", {})[episode_key] = {
                        "method": "M1",
                        "source_method": "M0",
                        "source_run_id": m0_run_id,
                        "source_trace": str(m0_trace),
                    }
                    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            if diagnostic_matrix:
                env["STAGE2_DIAGNOSTIC_MODE"] = method
                env["PMG_PERCEPTION_SOURCE"] = {
                    "M0": "none",
                    "M1": "oracle",
                    "M2": "predicted",
                    "M3": "oracle",
                }[method]
                env["PMG_ACTION_ENABLED"] = "1" if method in {"M1", "M2"} else "0"
            # Long outdoor_small episodes can exhaust the RTX pool on a
            # 32-GB compatibility GPU. Keep the episode, model, evaluator,
            # and PMG logic unchanged while lowering only the compatibility
            # render footprint for the known long-run/OOM cases. The same
            # setting is applied to every method, preserving paired fairness.
            if index in {51, 54, 55, 56, 57, 58, 59, 60}:
                env["TICVLA_RENDER_WIDTH"] = "96"
                env["TICVLA_RENDER_HEIGHT"] = "72"
                # Some outdoor_small layers report stale optional references
                # after the usable stage has opened. Allow the compatibility
                # runner to proceed without waiting on those non-critical
                # references; robot assets remain strictly gated.
                env["TICVLA_SKIP_SCENE_LOAD_WAIT"] = "1"
            log_path = log_root / f"{run_id}.log"
            progress["runs"][run_id] = {
                "status": "running",
                "result_class": "development_smoke" if smoke_frames is not None else "formal",
                "smoke_frames": smoke_frames,
                "started": datetime.now(timezone.utc).isoformat(),
                "official_num_people": official_people,
                "runtime_num_people": 0,
                "config": str(config_path),
            }
            status_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            started = time.time()
            command = ["bash", "scripts/08_run_dynanav_baseline.sh", str(config_path)]
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=project_main / "autodl",
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            valid = valid_result(root, run_id, method, episode_key, official_people)
            progress["runs"][run_id].update(
                {
                    "status": "passed" if completed.returncode == 0 and valid else "failed",
                    "exit_code": completed.returncode,
                    "finished": datetime.now(timezone.utc).isoformat(),
                    "wall_seconds": time.time() - started,
                    "result": str(result_path),
                    "log": str(log_path),
                    "valid_result": valid,
                }
            )
            status_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            print(json.dumps({"run_id": run_id, **progress["runs"][run_id]}, sort_keys=True), flush=True)

    progress["finished"] = datetime.now(timezone.utc).isoformat()
    progress["intended_processes"] = len(official_episode_keys) * len(selected_methods)
    progress["passed"] = sum(item.get("status") == "passed" for item in progress["runs"].values())
    progress["failed"] = sum(item.get("status") == "failed" for item in progress["runs"].values())
    status_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    if progress["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
