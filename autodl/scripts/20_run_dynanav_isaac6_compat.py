#!/usr/bin/env python3
"""Run one DynaNav episode on Isaac Sim 6 without the removed IRA 0.x API."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


# Isaac Sim's launcher may not preserve the caller's Python search path.  The
# compatibility runner lives two directories below the project package, so it
# can always add that package root without depending on shell state.
PROJECT_MAIN = Path(
    os.environ.get("PROJECT_MAIN", str(Path(__file__).resolve().parents[2]))
).resolve()
if str(PROJECT_MAIN) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_MAIN))


def _env_flag(*names: str) -> bool:
    """Return true when any named environment flag is explicitly enabled."""
    truthy = {"1", "true", "yes", "on"}
    return any(str(os.environ.get(name, "")).strip().lower() in truthy for name in names)


def parity_fixed_horizon_enabled() -> bool:
    """Keep paired diagnostic runs at a common horizon after first success.

    This is intentionally opt-in.  Legacy Stage 2 runs retain the benchmark's
    normal early-success termination unless one of the explicit parity flags
    is set by the caller.
    """
    return _env_flag("PMG_PARITY_FIXED_HORIZON", "STAGE2_PARITY_FIXED_HORIZON")


def parity_deterministic_enabled() -> bool:
    """Enable stricter Torch/CUDA determinism only for parity experiments."""
    return parity_fixed_horizon_enabled() or _env_flag(
        "PMG_PARITY_DETERMINISTIC", "STAGE2_PARITY_DETERMINISTIC"
    )


def stuck_recovery_disabled_enabled() -> bool:
    """Return whether the timing-sensitive legacy backup heuristic is off."""
    return _env_flag("STAGE2_DISABLE_STUCK_RECOVERY")


def strict_parity_contract_enabled() -> bool:
    """Return true only when all required paired-replay guards are active."""
    return bool(
        parity_fixed_horizon_enabled()
        and parity_deterministic_enabled()
        and stuck_recovery_disabled_enabled()
    )


def configure_parity_determinism() -> bool:
    """Configure deterministic Torch/CUDA execution when explicitly requested.

    The cuBLAS workspace variable is set before importing Torch so CUDA kernels
    can honor it.  Unsupported deterministic operators are warnings rather
    than fatal errors because the released TIC-VLA checkpoint may use an
    operation without a deterministic implementation on a given driver.
    """
    if not parity_deterministic_enabled():
        return False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch

        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        # Highest precision avoids a TF32-vs-FP32 kernel choice changing the
        # first policy query across otherwise identical paired processes.
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
    except Exception as exc:
        print(f"[parity] deterministic Torch/CUDA setup unavailable: {exc}", flush=True)
    return True


def resolve_diagnostic_environment() -> dict[str, object]:
    """Resolve explicit M0-M3 diagnostic switches without changing legacy runs."""
    explicit = os.environ.get("STAGE2_DIAGNOSTIC_MODE")
    method = str(os.environ.get("PMG_METHOD", "M0")).upper()
    if explicit:
        project_main = os.environ.get("PROJECT_MAIN")
        if project_main and project_main not in os.sys.path:
            os.sys.path.insert(0, project_main)
        from pmg_nav.diagnostics import resolve_contract

        contract = resolve_contract(explicit)
        os.environ["STAGE2_DIAGNOSTIC_MODE"] = contract.mode
        os.environ["PMG_METHOD"] = contract.mode
        os.environ.setdefault("PMG_PERCEPTION_SOURCE", contract.perception_source)
        os.environ.setdefault("PMG_ACTION_ENABLED", "1" if contract.action_enabled else "0")
        if contract.mode == "M3":
            # Strict M3 is a computation-only parity condition: use the same
            # compact/simple-oracle candidate and route calculations as M1,
            # but keep the behavior gate disabled and do not publish the
            # resulting payload to its control environment.
            os.environ["PMG_M3_PARITY_PROFILE"] = "1"
            os.environ["STAGE2_M1_VARIANT"] = "simple_oracle"
            os.environ["PMG_GRAPH_MODE"] = "compact"
            os.environ.setdefault("PMG_MAX_GRAPH_NODES", "8")
            os.environ.setdefault("PMG_MAX_EDGES_PER_ANCHOR", "4")
            os.environ["PMG_ROUTE_PLANNER"] = "1"
            os.environ.setdefault("PMG_M3_PARITY_CONTROL_POLICY", "terminal")
            os.environ.setdefault("PMG_M3_PARITY_INJECT_CONFLICT", "0")
            if parity_fixed_horizon_enabled():
                # M3 must not perturb the live M0 callback schedule.  Its PMG
                # graph is replayed from recorded poses after the horizon.
                os.environ["PMG_M3_DEFER_LIVE"] = "1"
                os.environ["PMG_CONTROL_POLICY"] = "policy_only"
            # Stage 2 parity is intentionally adapter-free.  Learned/vector
            # adapters belong to the Stage 3 RL bridge and would add a second
            # source of timing or control differences here.
            os.environ["PMG_ENABLE_GOAL_ADAPTER"] = "0"
            os.environ["PMG_ENABLE_VECTOR_ADAPTER"] = "0"
        return contract.to_dict()
    # Existing official v11 runs retain their historical oracle semantics.
    source = os.environ.get("PMG_PERCEPTION_SOURCE", "oracle" if method != "M0" else "none")
    action_enabled = os.environ.get("PMG_ACTION_ENABLED", "1" if method != "M0" else "0") == "1"
    os.environ.setdefault("PMG_PERCEPTION_SOURCE", source)
    os.environ.setdefault("PMG_ACTION_ENABLED", "1" if action_enabled else "0")
    return {
        "mode": method,
        "perception_source": source,
        "action_enabled": action_enabled,
        "pmg_built": method != "M0",
        "description": "legacy Stage 2 run (not the strict M0-M3 diagnostic contract)",
    }


def _finite_trace_value(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def read_action_trace(path: Path) -> dict[str, object]:
    """Summarize optional behavior-side action trace without failing an episode."""
    entries: list[dict] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                entries.append(value)
    outputs = [item for item in entries if item.get("kind") == "output"]
    inputs = [item for item in entries if item.get("kind") == "input"]
    inferences = [item for item in entries if item.get("kind") == "inference"]
    inference_latencies = [
        value
        for value in (_finite_trace_value(item.get("inference_latency_s")) for item in inferences)
        if value is not None
    ]
    output_values = [[item.get("v_cmd"), item.get("w_cmd")] for item in outputs]
    input_checksums = [item.get("checksum") for item in inputs if item.get("checksum")]
    policy_input_checksums = [
        item.get("policy_input_checksum")
        for item in inputs
        if item.get("policy_input_checksum")
    ]
    pmg_payload_checksums = [
        item.get("pmg_payload_checksum")
        for item in inputs
        if item.get("pmg_payload_checksum")
    ]
    greedy_sampling_flags = [
        bool(item.get("parity_greedy_sampling"))
        for item in inputs
        if "parity_greedy_sampling" in item
    ]
    override_outputs = [item for item in outputs if bool(item.get("pmg_override", item.get("recovery", False)))]
    source_counts: dict[str, int] = {}
    scope_counts: dict[str, int] = {}
    scgp_reason_counts: dict[str, int] = {}
    for item in outputs:
        source = str(item.get("pmg_control_source", "policy"))
        scope = str(item.get("pmg_gate_scope", "policy"))
        source_counts[source] = source_counts.get(source, 0) + 1
        scope_counts[scope] = scope_counts.get(scope, 0) + 1
        if bool(item.get("pmg_scgp_enabled")):
            reason = str(item.get("pmg_scgp_reason", "missing"))
            scgp_reason_counts[reason] = scgp_reason_counts.get(reason, 0) + 1
    action_count = len(outputs)
    override_count = len(override_outputs)
    # Prefer the per-tick value emitted by the behavior, but derive the
    # fraction as a compatibility fallback for older traces.
    last_fraction = None
    if outputs and outputs[-1].get("pmg_override_fraction") is not None:
        try:
            last_fraction = float(outputs[-1]["pmg_override_fraction"])
        except (TypeError, ValueError):
            last_fraction = None
    control_candidates = [item.get("pmg_control_candidate") for item in outputs if item.get("pmg_control_candidate") is not None]
    semantic_candidates = [item.get("pmg_semantic_candidate") for item in outputs if item.get("pmg_semantic_candidate") is not None]
    # ``pmg_route_enabled`` is emitted on every output tick, including M0 and
    # strict-M3 bypass ticks where it is explicitly false.  Counting field
    # presence here used to report every policy tick as a route output and
    # made a bypass look as if it had published a route.  A route output is
    # only a tick with an enabled PMG route; deferred/disabled ticks remain
    # available through the PMG summary and their explicit reason fields.
    route_outputs = [item for item in outputs if bool(item.get("pmg_route_enabled"))]
    route_deferred_outputs = [
        item
        for item in outputs
        if str(item.get("pmg_route_reason", "")).startswith("deferred")
    ]
    alignment_outputs = [item for item in outputs if bool(item.get("pmg_route_alignment_hold"))]
    route_fallback_outputs = [item for item in outputs if bool(item.get("pmg_route_stall_policy_fallback"))]
    heading_errors = [
        value
        for value in (
            _finite_trace_value(item.get("pmg_route_heading_error_rad"))
            for item in outputs
        )
        if value is not None
    ]
    return {
        "path": str(path),
        "entries": len(entries),
        "action_output_count": len(outputs),
        "action_input_count": len(inputs),
        "inference_event_count": len(inferences),
        "simulation_frame_first": next(
            (item.get("simulation_frame") for item in entries if item.get("simulation_frame") is not None),
            None,
        ),
        "simulation_frame_last": next(
            (item.get("simulation_frame") for item in reversed(entries) if item.get("simulation_frame") is not None),
            None,
        ),
        "inference_latency_mean_s": (
            sum(inference_latencies) / max(1, len(inference_latencies))
            if inference_latencies
            else None
        ),
        "inference_latency_max_s": (
            max(inference_latencies)
            if inference_latencies
            else None
        ),
        "action_output_checksum": hashlib.sha256(json.dumps(output_values, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest() if outputs else None,
        "action_input_checksum": hashlib.sha256(json.dumps(input_checksums, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest() if input_checksums else None,
        "policy_input_checksum": hashlib.sha256(
            json.dumps(policy_input_checksums, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest() if policy_input_checksums else None,
        # This is a sequence digest over per-query PMG payload hashes.  It is
        # intentionally separate from ``policy_input_checksum``: the latter
        # is the parity field for M0/M3, while this field audits structured
        # PMG data that may legitimately differ in M1/M2.
        "pmg_payload_checksum": hashlib.sha256(
            json.dumps(pmg_payload_checksums, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest() if pmg_payload_checksums else None,
        "policy_input_checksum_count": len(policy_input_checksums),
        "policy_input_checksum_unique_count": len(set(policy_input_checksums)),
        "pmg_payload_checksum_count": len(pmg_payload_checksums),
        "pmg_payload_checksum_unique_count": len(set(pmg_payload_checksums)),
        "parity_greedy_sampling": bool(greedy_sampling_flags)
        and all(greedy_sampling_flags),
        "parity_greedy_sampling_count": sum(greedy_sampling_flags),
        "recovery_trigger_count": sum(bool(item.get("recovery")) for item in outputs),
        "pmg_override_count": override_count,
        "pmg_override_fraction": (
            last_fraction if last_fraction is not None else override_count / max(1, action_count)
        ),
        "pmg_terminal_override_count": sum(
            bool(item.get("pmg_override")) and str(item.get("pmg_gate_scope", "")) == "terminal"
            for item in outputs
        ),
        "pmg_global_override_count": sum(
            bool(item.get("pmg_override")) and str(item.get("pmg_gate_scope", "")) == "global"
            for item in outputs
        ),
        "pmg_detour_override_count": sum(
            bool(item.get("pmg_override")) and str(item.get("pmg_control_source", "")) == "pmg_detour"
            for item in outputs
        ),
        "policy_action_count": sum(
            not bool(item.get("pmg_override", item.get("recovery", False)))
            for item in outputs
        ),
        "control_source_counts": source_counts,
        "control_scope_counts": scope_counts,
        "pmg_scgp_enabled": any(bool(item.get("pmg_scgp_enabled")) for item in outputs),
        "pmg_scgp_allow_count": sum(
            bool(item.get("pmg_scgp_enabled")) and bool(item.get("pmg_scgp_allow"))
            for item in outputs
        ),
        "pmg_scgp_deny_count": sum(
            bool(item.get("pmg_scgp_enabled")) and not bool(item.get("pmg_scgp_allow"))
            for item in outputs
        ),
        "pmg_scgp_reason_counts": scgp_reason_counts,
        "pmg_control_candidate_count": len(control_candidates),
        "pmg_control_candidate_first": control_candidates[0] if control_candidates else None,
        "pmg_control_candidate_last": control_candidates[-1] if control_candidates else None,
        "pmg_semantic_candidate_first": semantic_candidates[0] if semantic_candidates else None,
        "pmg_semantic_candidate_last": semantic_candidates[-1] if semantic_candidates else None,
        "pmg_route_enabled": any(bool(item.get("pmg_route_enabled")) for item in route_outputs),
        "pmg_route_valid_count": sum(bool(item.get("pmg_route_valid")) for item in route_outputs),
        "pmg_route_output_count": len(route_outputs),
        "pmg_route_deferred_output_count": len(route_deferred_outputs),
        "pmg_route_alignment_hold_count": len(alignment_outputs),
        "pmg_route_alignment_motion_violation_count": sum(
            abs(_finite_trace_value(item.get("v_cmd")) or 0.0) > 1e-6
            for item in alignment_outputs
        ),
        "pmg_route_stall_policy_fallback_count": len(route_fallback_outputs),
        "pmg_route_heading_error_max_abs_rad": (
            max(abs(value) for value in heading_errors) if heading_errors else None
        ),
        "pmg_route_reason_last": (
            str(route_outputs[-1].get("pmg_route_reason"))
            if route_outputs and route_outputs[-1].get("pmg_route_reason") is not None
            else None
        ),
        "pmg_semantic_candidate_distance_m": (
            _finite_trace_value(outputs[-1].get("pmg_semantic_candidate_distance_m"))
            if outputs else None
        ),
        "pmg_control_candidate_distance_m": (
            _finite_trace_value(outputs[-1].get("pmg_control_candidate_distance_m"))
            if outputs else None
        ),
        "control_policy": next(
            (str(item.get("control_policy")) for item in reversed(outputs) if item.get("control_policy") is not None),
            None,
        ),
        "pmg_vector_consumed": any(bool(item.get("pmg_vector_consumed")) for item in inputs),
    }

import yaml


def load_pmg_runtime():
    project_main = os.environ.get("PROJECT_MAIN")
    if project_main and project_main not in os.sys.path:
        os.sys.path.insert(0, project_main)
    from pmg_nav.runtime import PMGEpisodeState
    return PMGEpisodeState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-script", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--episode-name")
    parser.add_argument(
        "--memory-hook",
        type=Path,
        help=(
            "optional JSON/JSONL memory-dependent episode fixture; feeds PMG "
            "history while leaving the TIC-VLA/upstream evaluator unchanged"
        ),
    )
    parser.add_argument(
        "--memory-hook-episode",
        help="episode id inside --memory-hook (defaults to the first fixture row)",
    )
    parser.add_argument("--experience", type=Path)
    parser.add_argument("--warmup-frames", type=int, default=200)
    parser.add_argument(
        "--behavior-backend",
        choices=("manual_behavior", "usd_scripting"),
        default=os.environ.get("TICVLA_BEHAVIOR_BACKEND", "manual_behavior"),
        help=(
            "manual_behavior imports and drives the generated behavior explicitly; "
            "usd_scripting retains the legacy automatic USD scripting backend"
        ),
    )
    parser.add_argument(
        "--smoke-frames",
        "--manual-behavior-smoke-frames",
        dest="smoke_frames",
        type=int,
        default=int(os.environ.get("TICVLA_SMOKE_FRAMES", "0")),
        help="cap the episode to this many navigation frames (0 keeps the configured timeout)",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be non-negative")
    if args.smoke_frames < 0:
        parser.error("--smoke-frames must be non-negative")
    return args


def load_benchmark_module(path: Path):
    spec = importlib.util.spec_from_file_location("dynanav_stage1_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load DynaNav benchmark module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def new_behavior_provenance(
    script_path: Path,
    backend: str,
    robot_prim_path: str | None = None,
) -> dict[str, object]:
    """Create a JSON-serializable audit record before loading model code."""
    resolved = script_path.resolve()
    manual = backend == "manual_behavior"
    return {
        "backend": backend,
        "automatic_usd_attachment": not manual,
        "lifecycle_driver": (
            "explicit_manual_loop" if manual else "kit_usd_script_manager"
        ),
        "script_path": str(resolved),
        "script_sha256": (
            hashlib.sha256(resolved.read_bytes()).hexdigest()
            if resolved.is_file()
            else None
        ),
        "module_name": None,
        "class_name": "NovaCarterTICVLA",
        "robot_prim_path": robot_prim_path,
        "constructor_calls": 0 if manual else None,
        "on_init_calls": 0 if manual else None,
        "on_init_source": None,
        "on_play_calls": 0 if manual else None,
        "on_update_calls": 0 if manual else None,
        "on_stop_calls": 0 if manual else None,
        "on_destroy_calls": 0 if manual else None,
        "first_update_frame": None,
        "last_update_frame": None,
        "delta_time_source": "timeline_difference_with_30hz_fallback",
        "pmg_payload_publish_count": 0,
        "pmg_m3_deferred_live": bool(
            _env_flag("PMG_M3_DEFER_LIVE")
            and str(os.environ.get("PMG_METHOD", "M0")).upper() == "M3"
        ),
        "pmg_deferred_observation_count": 0,
        "pmg_deferred_payload_count": 0,
        "robot_usd_scripts_removed": [],
        "scripting_extension_active_during_manual_run": None,
        "scripting_extension_active_during_run": None,
        "configured_timeout_frames": None,
        "effective_timeout_frames": None,
        "smoke_mode": False,
        "parity_fixed_horizon": bool(parity_fixed_horizon_enabled()),
        "deterministic_torch_cuda": bool(parity_deterministic_enabled()),
        "stuck_recovery_disabled": bool(stuck_recovery_disabled_enabled()),
        "strict_parity_contract": bool(strict_parity_contract_enabled()),
        "first_goal_reached_frame": None,
        "errors": [],
    }


def _record_manual_behavior_error(
    provenance: dict[str, object], phase: str, exc: BaseException, *, fatal: bool
) -> None:
    error = {
        "phase": phase,
        "type": type(exc).__name__,
        "message": str(exc),
        "fatal": bool(fatal),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }
    errors = provenance.setdefault("errors", [])
    if isinstance(errors, list):
        errors.append(error)
    print(
        f"[manual_behavior] {phase} failed "
        f"({type(exc).__name__}): {exc}",
        flush=True,
    )
    print(error["traceback"], end="" if error["traceback"].endswith("\n") else "\n", flush=True)


def load_manual_behavior(
    script_path: Path,
    robot_prim_path: str,
    provenance: dict[str, object],
):
    """Import and initialize the generated behavior without a USD script API.

    The deployed Kit 107 ``BehaviorScript`` constructor calls ``on_init``
    unconditionally.  Never call it a second time here: that can construct a
    second robot wrapper and load the TIC-VLA model twice.
    """
    if not script_path.is_file():
        raise FileNotFoundError(f"generated TIC-VLA behavior is missing: {script_path}")
    module_name = f"ticvla_manual_behavior_{hashlib.sha256(str(script_path).encode()).hexdigest()[:12]}"
    provenance["module_name"] = module_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load generated TIC-VLA behavior: {script_path}")
    module = importlib.util.module_from_spec(spec)
    # Some generated behavior dependencies inspect sys.modules while their
    # module body is running.  Register the unique, per-script name first.
    os.sys.modules[module_name] = module
    try:
        from pxr import Sdf

        spec.loader.exec_module(module)
        behavior_class = getattr(module, "NovaCarterTICVLA", None)
        if behavior_class is None or not isinstance(behavior_class, type):
            raise RuntimeError(
                f"generated behavior has no NovaCarterTICVLA class: {script_path}"
            )
        behavior = behavior_class(Sdf.Path(robot_prim_path))
        provenance["constructor_calls"] = 1
        provenance["on_init_calls"] = 1
        provenance["on_init_source"] = "BehaviorScript_constructor"
        return behavior
    except Exception:
        os.sys.modules.pop(module_name, None)
        raise


def remove_robot_usd_behavior_scripts(stage, robot_prim_path: str) -> list[str]:
    """Remove only robot-attached scripts so the manual instance is unique."""
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim or not robot_prim.IsValid():
        raise RuntimeError(f"robot prim is missing: {robot_prim_path}")
    attr = robot_prim.GetAttribute("omni:scripting:scripts")
    if not attr or not attr.IsValid():
        return []
    value = attr.Get()
    existing = [str(getattr(item, "path", item)) for item in (value or [])]
    if existing:
        # Author an empty AssetArray on this runtime stage.  Do not remove the
        # schema/property: keeping it typed avoids the Kit 107 empty-type bug.
        attr.Set([])
        print(
            "[manual_behavior] cleared robot USD behavior scripts: "
            + ", ".join(existing),
            flush=True,
        )
    return existing


def attach_robot_usd_behavior_script(
    stage,
    robot_prim_path: str,
    script_path: Path,
) -> None:
    """Retain the previous automatic USD scripting backend as an option."""
    from pxr import Sdf

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim or not robot_prim.IsValid():
        raise RuntimeError(f"robot prim is missing: {robot_prim_path}")
    try:
        from OmniScriptingSchema import OmniScriptingAPI

        api = OmniScriptingAPI.Apply(robot_prim)
        attr = api.GetScriptsAttr()
    except Exception:
        attr = robot_prim.GetAttribute("omni:scripting:scripts")
    if not attr or not attr.IsValid() or not attr.GetTypeName():
        if attr and attr.IsValid():
            robot_prim.RemoveProperty(attr.GetName())
        attr = robot_prim.CreateAttribute(
            "omni:scripting:scripts",
            Sdf.ValueTypeNames.AssetArray,
            custom=False,
        )
    if attr.GetTypeName() != Sdf.ValueTypeNames.AssetArray:
        raise RuntimeError(
            "unexpected type for omni:scripting:scripts: "
            f"{attr.GetTypeName()}"
        )
    attr.Set([Sdf.AssetPath(str(script_path))])
    print(
        f"[usd_scripting] attached generated behavior to {robot_prim_path}: "
        f"{script_path}",
        flush=True,
    )


def record_deferred_m3_observation(
    runner,
    provenance: dict[str, object],
    frame_id: int,
    position: tuple[float, float, float],
) -> None:
    """Record one strict-M3 pose without touching the live policy path.

    Strict M3 is a no-authority ablation.  Even the small environment writes
    used to mark a deferred PMG payload can change Replicator/manual callback
    cadence around an image capture.  Keep the live callback side-effect free;
    :func:`finalize_deferred_m3_pmg` replays these poses after the fixed
    horizon.
    """
    deferred = getattr(runner, "_pmg_deferred_observations", None)
    if deferred is None:
        deferred = []
        runner._pmg_deferred_observations = deferred
    deferred.append((int(frame_id), tuple(map(float, position))))
    provenance["pmg_deferred_observation_count"] = len(deferred)


def publish_pmg_action_payload(runner, provenance: dict[str, object]) -> None:
    """Publish one PMG frame to the generated behavior's environment API."""
    if getattr(runner, "_pmg_state", None) is None or runner._robot_position is None:
        return
    position = tuple(map(float, runner._robot_position))
    # Strict M3 is a no-authority ablation.  Running the complete graph and
    # route calculation inside the live callback can still change Isaac's
    # RGB/query schedule even when no payload is published.  Record the same
    # robot poses cheaply and replay them after the fixed-horizon episode so
    # the PMG artifact remains available without becoming a timing side-channel.
    if (
        _env_flag("PMG_M3_DEFER_LIVE")
        and str(os.environ.get("PMG_METHOD", "M0")).upper() == "M3"
        and bool(getattr(runner._pmg_state, "m3_parity_profile", False))
    ):
        # Do not call PMG or mutate TICVLA_PMG_* during a live strict-M3
        # callback.  M0 has no corresponding work, so even these apparently
        # harmless writes can perturb image capture/query cadence.  The
        # recorded pose is replayed only after the fixed horizon.
        record_deferred_m3_observation(
            runner,
            provenance,
            int(runner._simulation_frame_count),
            position,
        )
        return
    runner._pmg_state.update_from_robot(runner._simulation_frame_count, position)
    diagnostic_only = bool(
        getattr(runner._pmg_state, "m3_parity_profile", False)
        and not getattr(runner._pmg_state, "action_enabled", True)
    )
    payload = runner._pmg_state.action_payload(
        position,
        diagnostic_only=diagnostic_only,
    )
    for key in (
        "TICVLA_PMG_GOAL_X", "TICVLA_PMG_GOAL_Y", "TICVLA_PMG_GOAL_Z",
        "TICVLA_PMG_GOAL_DISTANCE", "TICVLA_PMG_GOAL_YAW",
        "TICVLA_PMG_SCORE", "TICVLA_PMG_HARD_VALID", "TICVLA_PMG_STABLE",
        "TICVLA_PMG_WAYPOINT_X", "TICVLA_PMG_WAYPOINT_Y", "TICVLA_PMG_WAYPOINT_Z",
        "TICVLA_PMG_WAYPOINT_INDEX", "TICVLA_PMG_ROUTE_LENGTH",
        "TICVLA_PMG_ROUTE_ENABLED", "TICVLA_PMG_ROUTE_VALID", "TICVLA_PMG_ROUTE_REASON",
        "TICVLA_PMG_SCGP_ENABLED", "TICVLA_PMG_SCGP_ALLOW", "TICVLA_PMG_SCGP_MODE",
        "TICVLA_PMG_SCGP_REASON", "TICVLA_PMG_SCGP_LEASE_UNTIL",
        "TICVLA_PMG_SCGP_STABLE_COUNT", "TICVLA_PMG_SCGP_NO_PROGRESS_TICKS",
        "TICVLA_PMG_DIAGNOSTIC_ONLY",
    ):
        os.environ.pop(key, None)
    # M3 parity deliberately computes/serializes the M1 payload but keeps it
    # out of the behavior environment.  This guarantees policy-only action
    # output while preserving candidate/route timing telemetry in the PMG
    # artifact.
    if payload is not None and not diagnostic_only:
        os.environ.update({str(key): str(value) for key, value in payload.items()})
    else:
        os.environ.update({
            "TICVLA_PMG_ROUTE_VALID": "0",
            "TICVLA_PMG_ROUTE_ENABLED": "0",
            "TICVLA_PMG_ROUTE_REASON": "no_pmg_payload",
            "TICVLA_PMG_SCGP_ENABLED": "0",
            "TICVLA_PMG_SCGP_ALLOW": "1",
            "TICVLA_PMG_SCGP_MODE": "disabled",
            "TICVLA_PMG_SCGP_REASON": "no_pmg_payload",
            "TICVLA_PMG_SCGP_LEASE_UNTIL": "-1",
            "TICVLA_PMG_SCGP_STABLE_COUNT": "0",
            "TICVLA_PMG_SCGP_NO_PROGRESS_TICKS": "0",
            "TICVLA_PMG_SCORE": "0",
            "TICVLA_PMG_HARD_VALID": "0",
            "TICVLA_PMG_STABLE": "0",
        })
        if diagnostic_only:
            os.environ["TICVLA_PMG_DIAGNOSTIC_ONLY"] = "1"
    provenance["pmg_payload_publish_count"] = int(
        provenance.get("pmg_payload_publish_count", 0)
    ) + 1
    if diagnostic_only:
        provenance["pmg_diagnostic_payload_count"] = int(
            provenance.get("pmg_diagnostic_payload_count", 0)
        ) + 1


def finalize_deferred_m3_pmg(runner, provenance: dict[str, object]) -> None:
    """Replay deferred M3 observations after live policy execution."""
    state = getattr(runner, "_pmg_state", None)
    if state is None or not (
        _env_flag("PMG_M3_DEFER_LIVE")
        and str(os.environ.get("PMG_METHOD", "M0")).upper() == "M3"
        and bool(getattr(state, "m3_parity_profile", False))
    ):
        return
    deferred = list(getattr(runner, "_pmg_deferred_observations", ()))
    if not deferred:
        return
    for frame_id, position in deferred:
        state.update_from_robot(int(frame_id), tuple(map(float, position)))
    # One final diagnostic-only serialization is enough to materialize the
    # candidate/constraint summary.  It runs after the fixed horizon and can
    # never influence a TIC-VLA callback or action.
    state.action_payload(tuple(map(float, deferred[-1][1])), diagnostic_only=True)
    provenance["pmg_deferred_observation_count"] = len(deferred)
    provenance["pmg_deferred_payload_count"] = 1


async def run_navigation_with_manual_behavior(
    runner,
    timeout_frames: int,
    behavior,
    provenance: dict[str, object],
) -> bool:
    """Run the upstream benchmark loop while explicitly driving behavior."""
    import omni.kit.app
    import omni.replicator.core as rep
    import omni.timeline

    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    original_timecode = timeline.get_time_codes_per_second()
    fixed_delta_time = 1.0 / 30.0
    success = False
    pending_error: BaseException | None = None
    fixed_horizon = parity_fixed_horizon_enabled()
    first_goal_reached_frame: int | None = None
    defer_m3_live = bool(
        _env_flag("PMG_M3_DEFER_LIVE")
        and str(os.environ.get("PMG_METHOD", "M0")).upper() == "M3"
        and bool(
            getattr(getattr(runner, "_pmg_state", None), "m3_parity_profile", False)
        )
    )
    provenance["parity_fixed_horizon"] = bool(fixed_horizon)
    provenance["deterministic_torch_cuda"] = bool(
        parity_deterministic_enabled()
    )
    try:
        timeline.set_time_codes_per_second(30)
        timeline.commit_silently()
        await app.next_update_async()
        # Replicator must be armed before the timeline starts.  In the manual
        # lifecycle there is no USD behavior callback to arm capture on the
        # play event; enabling it after ``timeline.play()`` leaves the RGB
        # annotators permanently empty and silently produces zero policy
        # queries.
        rep.orchestrator.set_capture_on_play(True)
        if not timeline.is_playing():
            timeline.play()
            # Kit applies ``timeline.play()`` on the next app tick.  The
            # manual behavior checks ``timeline.is_playing()`` at the top of
            # ``on_update``; invoking it before this tick makes the behavior
            # finalize at frame 1 and suppress capture, inference, and traces.
            await app.next_update_async()
            if not timeline.is_playing():
                raise RuntimeError(
                    "manual behavior timeline did not enter playing state "
                    "after timeline.play()"
                )

        try:
            provenance["on_play_calls"] = int(provenance.get("on_play_calls", 0)) + 1
            behavior.on_play()
        except Exception as exc:
            _record_manual_behavior_error(provenance, "on_play", exc, fatal=True)
            raise RuntimeError("manual TIC-VLA behavior failed during on_play") from exc

        # Do not advance a playing simulation frame before the first explicit
        # on_update call.  The first app update below is paired with frame 1;
        # this keeps manual and automatic callback cadence auditable.

        runner._episode_start_frame = runner._simulation_frame_count
        try:
            previous_time = float(timeline.get_current_time())
        except Exception:
            previous_time = float(runner._simulation_frame_count) * fixed_delta_time

        for _ in range(timeout_frames):
            await runner._update_robot_position()
            runner._simulation_frame_count += 1
            frame = runner._simulation_frame_count
            if runner.debug_print and frame % 30 == 0:
                print(f"Simulation frame count: {frame}")
            os.environ["BENCHMARK_SIMULATION_FRAME_COUNT"] = str(frame)

            if runner._is_goal_reached():
                # In a paired replay, M0 must continue issuing policy queries
                # after the first success so M0/M3 have an identical fixed
                # frame/query horizon.  Preserve the first-success frame for
                # evaluation while keeping the normal early break in legacy
                # runs.
                success = True
                if first_goal_reached_frame is None:
                    first_goal_reached_frame = frame
                    provenance["first_goal_reached_frame"] = frame
                if not fixed_horizon:
                    break

            # Strict M3 must not do PMG work or mutate TICVLA_PMG_* before the
            # paired policy callback.  Record this pose after on_update so the
            # live callback has the same pre-query work as M0.
            deferred_position = (
                tuple(map(float, runner._robot_position))
                if defer_m3_live and runner._robot_position is not None
                else None
            )
            if not defer_m3_live:
                publish_pmg_action_payload(runner, provenance)
            try:
                current_time = float(timeline.get_current_time())
            except Exception:
                current_time = float(frame) * fixed_delta_time
            delta_time = current_time - previous_time
            if not math.isfinite(delta_time) or delta_time <= 0.0 or delta_time > 0.5:
                delta_time = fixed_delta_time
            previous_time = current_time
            try:
                provenance["on_update_calls"] = int(
                    provenance.get("on_update_calls", 0)
                ) + 1
                behavior.on_update(current_time, delta_time)
            except Exception as exc:
                _record_manual_behavior_error(
                    provenance, f"on_update(frame={frame})", exc, fatal=True
                )
                raise RuntimeError(
                    f"manual TIC-VLA behavior failed at simulation frame {frame}"
                ) from exc
            if provenance.get("first_update_frame") is None:
                provenance["first_update_frame"] = frame
            provenance["last_update_frame"] = frame

            if deferred_position is not None:
                record_deferred_m3_observation(
                    runner,
                    provenance,
                    frame,
                    deferred_position,
                )

            await app.next_update_async()
    except BaseException as exc:
        pending_error = exc
    finally:
        try:
            rep.orchestrator.set_capture_on_play(False)
        except Exception as exc:
            _record_manual_behavior_error(provenance, "disable_capture", exc, fatal=False)
        try:
            timeline.stop()
        except Exception as exc:
            _record_manual_behavior_error(provenance, "timeline_stop", exc, fatal=False)
        try:
            provenance["on_stop_calls"] = int(provenance.get("on_stop_calls", 0)) + 1
            behavior.on_stop()
        except Exception as exc:
            _record_manual_behavior_error(provenance, "on_stop", exc, fatal=False)
        try:
            provenance["on_destroy_calls"] = int(
                provenance.get("on_destroy_calls", 0)
            ) + 1
            behavior.on_destroy()
        except Exception as exc:
            _record_manual_behavior_error(provenance, "on_destroy", exc, fatal=False)
        try:
            timeline.set_time_codes_per_second(original_timecode)
            timeline.commit_silently()
            await app.next_update_async()
        except Exception as exc:
            _record_manual_behavior_error(
                provenance, "restore_timeline", exc, fatal=False
            )
    if pending_error is not None:
        raise pending_error
    return success


def select_episode(config: dict, requested: str | None) -> tuple[str, dict]:
    if requested:
        episode = config.get(requested)
        if not isinstance(episode, dict):
            raise KeyError(f"episode is missing from config: {requested}")
        return requested, episode
    for name, episode in config.items():
        if name not in {"seed", "navigation_method", "success_threshold"} and isinstance(episode, dict):
            return name, episode
    raise RuntimeError("benchmark config contains no episode")


def seed_episode(
    config: dict,
    episode_name: str,
    *,
    defer_torch: bool = False,
) -> int:
    """Set one method-independent RNG stream for a paired episode.

    Isaac Kit must own the first GPU/UVM initialization in the compatibility
    runtime.  Importing Torch and probing ``torch.cuda`` before
    ``SimulationApp`` can deadlock on the current 5090 image, so callers that
    are about to create Kit defer only Torch/CUDA setup and apply it after the
    app has started.  Python and NumPy RNGs remain seeded before Kit so the
    episode contract is unchanged.
    """
    try:
        episode_index = int(str(episode_name).rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        episode_index = 0
    seed = int(config.get("seed", 666)) + episode_index
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Configure deterministic CUDA/Torch behavior only after Kit has started
    # when ``defer_torch`` is requested.  This avoids a CUDA/UVM initialization
    # deadlock observed before SimulationApp construction on the RTX 5090
    # compatibility image.
    deterministic = parity_deterministic_enabled()
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    if not defer_torch:
        configure_torch_seed(seed, deterministic=deterministic)
    if deterministic:
        os.environ["STAGE2_PARITY_DETERMINISTIC_ACTIVE"] = "1"
    return seed


def configure_torch_seed(seed: int, *, deterministic: bool | None = None) -> None:
    """Apply Torch/CUDA parity settings after Isaac Kit initialization."""
    if deterministic is None:
        deterministic = configure_parity_determinism()
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception as exc:
        print(f"[parity] Torch/CUDA seed setup unavailable: {exc}", flush=True)
    if deterministic:
        os.environ["STAGE2_PARITY_DETERMINISTIC_ACTIVE"] = "1"


def resolve_memory_hook_fixture(
    config: dict,
    config_path: Path,
    explicit_path: Path | None,
    explicit_episode: str | None,
) -> tuple[Path | None, str | None]:
    """Resolve an optional fixture without changing ordinary benchmark runs."""
    raw_path = explicit_path
    if raw_path is None:
        configured = config.get("memory_hook_fixture")
        raw_path = Path(str(configured)) if configured else None
    if raw_path is None:
        return None, None
    if not raw_path.is_absolute():
        project_candidate = PROJECT_MAIN / raw_path
        config_candidate = config_path.parent / raw_path
        raw_path = next(
            (candidate for candidate in (project_candidate, config_candidate) if candidate.is_file()),
            project_candidate,
        )
    resolved = raw_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"memory-hook fixture is missing: {resolved}")
    configured_episode = config.get("memory_hook_episode")
    episode = explicit_episode or (str(configured_episode) if configured_episode else None)
    return resolved, episode


def wait_for_task(sim_app, task):
    while not task.done() and not sim_app.is_exiting():
        sim_app.update()
    return task.result()


def finite_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def validate_episode_artifact(result: dict) -> tuple[str, list[str]]:
    """Classify whether an episode result is safe for benchmark aggregation.

    Isaac compatibility failures used to be serialized as an ordinary failed
    episode with zero duration and a null navigation error.  That silently
    contaminated SR/error averages.  Keep the raw result for debugging, but
    require a complete finite episode before it can enter a matrix.
    """
    reasons: list[str] = []
    error = result.get("error")
    if error:
        reasons.append("runner_error")
    try:
        duration_frames = int(result.get("duration_frames", 0) or 0)
    except (TypeError, ValueError):
        duration_frames = 0
    if duration_frames <= 0:
        reasons.append("zero_duration_frames")
    if finite_or_none(result.get("duration")) is None or float(result.get("duration", 0.0) or 0.0) <= 0.0:
        reasons.append("zero_duration")
    if finite_or_none(result.get("navigation_error")) is None:
        reasons.append("missing_navigation_error")
    trace = result.get("action_trace")
    if isinstance(trace, dict) and int(trace.get("action_output_count", 0) or 0) <= 0:
        reasons.append("missing_action_trace")
    return ("valid" if not reasons else "invalid_artifact"), reasons


def aggregate_result(config_path: Path, result: dict) -> dict:
    artifact_status, artifact_reasons = validate_episode_artifact(result)
    success = bool(result.get("success", False))
    collision = float(result.get("collision", 0.0) or 0.0)
    human_collision = float(result.get("human_collision", 0.0) or 0.0)
    physical_collision = float(result.get("physical_collision", 0.0) or 0.0)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path),
        "compatibility_runner": "isaac_sim_6_without_ira_0x",
        "behavior_backend": result.get("behavior_backend"),
        "behavior_provenance": result.get("behavior_provenance"),
        "total_episodes": 1,
        "successful_episodes": int(success),
        "failed_episodes": int(not success),
        "success_rate_percent": 100.0 if success else 0.0,
        "artifact_status": artifact_status,
        "artifact_invalid_reasons": artifact_reasons,
        "average_metrics": {
            "navigation_error_m": finite_or_none(result.get("navigation_error")),
            "total_collision_rate_percent": collision * 100.0,
            "human_collision_rate_percent": human_collision * 100.0,
            "physical_contact_rate_percent": physical_collision * 100.0,
            "spl": float(result.get("spl", 0.0) or 0.0),
            "path_length_m": float(result.get("path_length", 0.0) or 0.0),
            "duration_s": float(result.get("duration", 0.0) or 0.0),
            "duration_frames": int(result.get("duration_frames", 0) or 0),
        },
        "episodes": [result],
    }


def main() -> None:
    args = parse_args()
    args.benchmark_script = args.benchmark_script.resolve()
    args.config = args.config.resolve()
    if args.experience is not None:
        args.experience = args.experience.resolve()
        if not args.experience.is_file():
            raise FileNotFoundError(f"Isaac Sim experience is missing: {args.experience}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    episode_name, episode = select_episode(config, args.episode_name)
    # Do not import/probe Torch CUDA before SimulationApp.  The RTX 5090
    # compatibility image can deadlock in CUDA/UVM initialization before Kit
    # has established its device context; the deferred setup is applied just
    # after SimulationApp construction below.
    episode_seed = seed_episode(config, episode_name, defer_torch=True)
    memory_hook_path, memory_hook_episode = resolve_memory_hook_fixture(
        config,
        args.config,
        args.memory_hook,
        args.memory_hook_episode,
    )
    if memory_hook_path is not None:
        os.environ["PMG_MEMORY_HOOK_PATH"] = str(memory_hook_path)
        if memory_hook_episode:
            os.environ["PMG_MEMORY_HOOK_EPISODE"] = str(memory_hook_episode)
        else:
            os.environ.pop("PMG_MEMORY_HOOK_EPISODE", None)
    else:
        os.environ.pop("PMG_MEMORY_HOOK_PATH", None)
        os.environ.pop("PMG_MEMORY_HOOK_EPISODE", None)
    diagnostic_contract = resolve_diagnostic_environment()
    os.environ["BENCHMARK_RUN_ID"] = args.run_id
    os.environ["NAVIGATION_METHOD"] = str(config.get("navigation_method", "ticvla"))
    os.environ["TICVLA_INSTRUCTION"] = str(episode.get("instruction", ""))
    os.environ["TICVLA_EPISODE_NAME"] = episode_name
    os.environ["TICVLA_EPISODE_SEED"] = str(episode_seed)
    goal = episode.get("goal", [0.0, 0.0, 0.0])
    os.environ["TICVLA_GOAL_X"] = str(float(goal[0]))
    os.environ["TICVLA_GOAL_Y"] = str(float(goal[1]))
    os.environ["TICVLA_GOAL_Z"] = str(float(goal[2] if len(goal) > 2 else 0.0))
    # Prevent route fields from a prior in-process episode from leaking into
    # a fresh baseline/diagnostic run.  The evaluator goal remains confined to
    # the benchmark runner and is never copied into PMG route state.
    for _route_key in (
        "TICVLA_PMG_WAYPOINT_X", "TICVLA_PMG_WAYPOINT_Y", "TICVLA_PMG_WAYPOINT_Z",
        "TICVLA_PMG_WAYPOINT_INDEX", "TICVLA_PMG_ROUTE_LENGTH",
    ):
        os.environ.pop(_route_key, None)
    os.environ["TICVLA_PMG_ROUTE_VALID"] = "0"
    os.environ["TICVLA_PMG_ROUTE_ENABLED"] = "0"
    os.environ["TICVLA_PMG_ROUTE_REASON"] = "episode_start"
    trace_path = Path(os.environ.get(
        "PMG_ACTION_TRACE_PATH",
        str(Path.cwd() / "benchmark_results" / args.run_id / "action_trace.jsonl"),
    ))
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.unlink(missing_ok=True)
    os.environ["PMG_ACTION_TRACE_PATH"] = str(trace_path)
    os.environ["PMG_DIAGNOSTIC_CONTRACT"] = json.dumps(diagnostic_contract, sort_keys=True)
    os.environ["TICVLA_BEHAVIOR_BACKEND"] = args.behavior_backend
    behavior_script_path = Path(os.environ.get(
        "TICVLA_BEHAVIOR_SCRIPT",
        str(args.benchmark_script.parent / "behavior" / "nova_carter_test_ticvla.py"),
    )).resolve()
    if not behavior_script_path.is_file():
        raise FileNotFoundError(
            f"generated Nova Carter behavior is missing: {behavior_script_path}"
        )
    os.environ["TICVLA_BEHAVIOR_SCRIPT"] = str(behavior_script_path)
    behavior_provenance = new_behavior_provenance(
        behavior_script_path,
        args.behavior_backend,
    )
    behavior_provenance["memory_hook"] = (
        {
            "path": str(memory_hook_path),
            "episode_id": memory_hook_episode,
            "instruction_match_required": bool(
                config.get("memory_hook_strict_instruction", True)
            ),
        }
        if memory_hook_path is not None
        else {"active": False}
    )

    from isaacsim import SimulationApp

    render_width = int(os.environ.get("TICVLA_RENDER_WIDTH", "640"))
    render_height = int(os.environ.get("TICVLA_RENDER_HEIGHT", "480"))
    descriptor_sets = int(os.environ.get("TICVLA_RTX_DESCRIPTOR_SETS", "20000"))
    app_config = {
        "renderer": "RayTracedLighting",
        "headless": True,
        "width": render_width,
        "height": render_height,
        "fast_shutdown": True,
        "extra_args": [f"--/rtx/descriptorSets={descriptor_sets}"],
    }
    experience = str(args.experience) if args.experience is not None else ""
    sim_app = SimulationApp(app_config, experience=experience)
    configure_torch_seed(episode_seed, deterministic=parity_deterministic_enabled())
    try:
        import carb
        carb.settings.get_settings().set("/rtx/descriptorSets", descriptor_sets)
    except Exception as exc:
        print(f"[compat] unable to increase RTX descriptor pool: {exc}", flush=True)
    result = None
    try:
        benchmark = load_benchmark_module(args.benchmark_script)

        async def run_episode() -> dict:
            nonlocal episode
            import omni.kit.app
            import omni.usd
            from pxr import Gf, Usd, UsdGeom
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                torch = None

            # Expand local DynaNav asset paths before the compatibility runner
            # opens the stage. The upstream runner normally does this internally.
            expanded_episode = benchmark._expand_env_value(episode)

            extension_manager = omni.kit.app.get_app().get_extension_manager()
            # Isaac Sim 5/Kit 107 exposes BehaviorScript through
            # ``omni.kit.scripting``.  ``omni.behavior.scripting.core`` was a
            # transient name used by an earlier image and is not installed
            # on the current RTX 5090 environment; requesting it makes Kit
            # block during dependency resolution before the behavior can
            # attach, yielding a zero-action trace.
            for extension in (
                "omni.kit.scripting",
                "isaacsim.replicator.behavior",
                "isaacsim.robot.wheeled_robots",
                "isaacsim.sensors.camera",
                "isaacsim.sensors.physics",
            ):
                extension_manager.set_extension_enabled_immediate(extension, True)
            await omni.kit.app.get_app().next_update_async()
            # Import the base class while the extension is healthy.  Static
            # office stages require the extension to remain disabled after
            # loading, but Python keeps this module available for the manual
            # behavior instance without restarting its broken stage manager.
            from omni.kit.scripting import BehaviorScript as _ManualBehaviorBase

            if _ManualBehaviorBase is None:
                raise RuntimeError("omni.kit.scripting.BehaviorScript is unavailable")

            runner = benchmark.BenchmarkRunner(
                sim_app,
                str(args.config),
                str(Path.cwd() / "crash_reports"),
                args.debug,
            )
            runner._benchmark_config = benchmark._expand_env_value(config)
            runner.current_episode = expanded_episode
            runner.episode_name = episode_name
            runner.episode_start_time = time.time()
            runner.episode_success = False
            runner._spawned_robot_prim_path = None
            runner._robot_type = None
            runner._robot_position = None
            runner._goal_position = None
            runner._robot_path = []
            runner._collision_detected = False
            runner._episode_start_position = None
            runner._simulation_frame_count = 0
            runner._episode_start_frame = 0
            runner._physical_collision_detected = False
            runner._contact_consecutive_frames = 0
            runner._max_contact_consecutive_frames = 0
            runner._goal_threshold = float(
                expanded_episode.get("success_threshold", config.get("success_threshold", 1.5))
            )
            runner._set_simulation_settings()

            context = omni.usd.get_context()
            episode = expanded_episode
            # Some released office USD layers contain authoring-time Python
            # hooks that open zenity when their optional assets are absent.
            # Load those static scenes with scripting disabled.  The manual
            # backend can keep the stage-script manager disabled; the optional
            # USD backend restores it only after the scene is usable.
            static_scene = str(episode["scene"]).lower().endswith(
                ("office.usd", "outdoor.usd", "outdoor_new.usd", "outdoor_small.usd")
            )
            # Isaac Sim 6's ScriptManager cannot always be unloaded after a
            # non-static stage has opened (warehouse scenes reproduce an
            # ``_on_stage_detach`` AttributeError). Keep the extension loaded
            # for those scenes by default; callers can opt into the old
            # behavior for a scene known to tolerate hard unloading.
            disable_nonstatic_scripting = bool(
                not static_scene
                and os.environ.get("TICVLA_DISABLE_NONSTATIC_SCRIPTING", "0")
                == "1"
            )
            if static_scene:
                import carb
                settings = carb.settings.get_settings()
                settings.set("/persistent/scripting/enablePythonScripting", False)
                settings.set("/persistent/scripting/allowPythonScripting", False)
                # Office assets may carry legacy authoring-time scripting hooks
                # that invoke GUI dialogs while the stage is opening. Isaac Sim
                # 6 can crash when this extension is disabled after
                # ``BehaviorScript`` has been imported, so settings disable
                # script execution by default and hard unloading is opt-in for
                # images where the extension manager is known to be stable.
                if os.environ.get("TICVLA_HARD_DISABLE_SCRIPTING", "0") == "1":
                    extension_manager.set_extension_enabled_immediate(
                        "omni.kit.scripting", False
                    )
            open_result = await context.open_stage_async(str(episode["scene"]))
            if isinstance(open_result, tuple):
                opened = bool(open_result[0])
            else:
                opened = bool(open_result)
            if not opened:
                raise RuntimeError(f"failed to open scene: {episode['scene']}")

            # A small subset of outdoor_small layers keeps reporting a
            # pending optional reference after the usable stage is open.  On
            # an explicitly requested compatibility retry, do not block the
            # benchmark on that non-critical reference; robot asset loading
            # below remains strictly gated.
            skip_static_wait = os.environ.get("TICVLA_SKIP_SCENE_LOAD_WAIT") == "1"
            if not skip_static_wait:
                deadline = time.monotonic() + 300.0
                while context.get_stage_loading_status()[2] and time.monotonic() < deadline:
                    await omni.kit.app.get_app().next_update_async()
                if context.get_stage_loading_status()[2]:
                    raise TimeoutError("scene references did not finish loading in 300 seconds")
            if static_scene and args.behavior_backend == "usd_scripting":
                extension_manager.set_extension_enabled_immediate(
                    "omni.kit.scripting", True
                )
                settings.set("/persistent/scripting/enablePythonScripting", True)
                settings.set("/persistent/scripting/allowPythonScripting", True)
                await omni.kit.app.get_app().next_update_async()
            if args.behavior_backend == "manual_behavior":
                # The base class was imported before stage load.  Disable the
                # manager before robot creation so a USD-authored script can
                # never instantiate a second copy beside the manual object.
                # On non-static Isaac Sim 6 scenes this unload is opt-in,
                # because ScriptManager teardown can abort the whole runner.
                if disable_nonstatic_scripting:
                    extension_manager.set_extension_enabled_immediate(
                        "omni.kit.scripting", False
                    )
                    await omni.kit.app.get_app().next_update_async()
                behavior_provenance[
                    "scripting_extension_active_during_manual_run"
                ] = not disable_nonstatic_scripting and not static_scene
                behavior_provenance[
                    "scripting_extension_active_during_run"
                ] = not disable_nonstatic_scripting and not static_scene
                print(
                    "[manual_behavior] automatic USD scripting manager "
                    + ("disabled" if disable_nonstatic_scripting else "kept loaded")
                    + " "
                    "before robot creation",
                    flush=True,
                )
            else:
                behavior_provenance["scripting_extension_active_during_run"] = True
            if static_scene and args.behavior_backend == "manual_behavior":
                print(
                    "[manual_behavior] keeping omni.kit.scripting manager "
                    "disabled after static-stage load",
                    flush=True,
                )
            for _ in range(args.warmup_frames):
                await omni.kit.app.get_app().next_update_async()

            robot_type = str(episode["robot_type"]).title()
            if robot_type == "Nova_Carter":
                robot_type = "Nova_Carter"
            runner._robot_type = robot_type
            runner._spawned_robot_prim_path = await runner._spawn_robot(
                robot_type,
                episode["start"],
                float(episode.get("start_yaw", 0.0)),
            )

            deadline = time.monotonic() + 300.0
            while context.get_stage_loading_status()[2] and time.monotonic() < deadline:
                await omni.kit.app.get_app().next_update_async()
            if context.get_stage_loading_status()[2]:
                raise TimeoutError("robot references did not finish loading in 300 seconds")

            stage = context.get_stage()
            third_person_path = (
                f"{runner._spawned_robot_prim_path}/chassis_link/third_person_view_cam"
            )
            if not stage.GetPrimAtPath(third_person_path).IsValid():
                camera = UsdGeom.Camera.Define(stage, third_person_path)
                xform = UsdGeom.Xformable(camera.GetPrim())
                xform.AddTranslateOp().Set(Gf.Vec3d(-4.0, 0.0, 2.5))
                xform.AddRotateXYZOp().Set(Gf.Vec3f(65.0, 0.0, -90.0))

            runner._contact_sensor_enabled = False
            runner._goal_position = benchmark.np.array(episode["goal"])
            runner._episode_start_position = benchmark.np.array(episode["start"])

            # Build and update PMG inside the episode loop. The annotation gate
            # selected USD identity/pose as the current oracle perception source.
            # The benchmark goal below remains evaluator-only: PMG runtime must
            # not use it to rank, synthesize, or filter candidates.
            pmg_method = str(os.environ.get("PMG_METHOD", "M0")).upper()
            if pmg_method != "M0":
                PMGEpisodeState = load_pmg_runtime()
                runner._pmg_state = PMGEpisodeState(
                    str(episode.get("instruction", "")),
                    pmg_method,
                    # The evaluator goal is intentionally not passed into PMG;
                    # success/navigation-error code below remains the sole
                    # consumer of benchmark ground truth.
                    goal_hint=None,
                )
                # The explicit diagnostic contract is set before stage
                # initialization so M2 cannot fall back to USD oracle data.
                runner._pmg_state.diagnostic_mode = str(
                    os.environ.get("STAGE2_DIAGNOSTIC_MODE", pmg_method)
                ).upper()
                runner._pmg_state.perception_source = str(
                    os.environ.get("PMG_PERCEPTION_SOURCE", runner._pmg_state.perception_source)
                ).lower()
                runner._pmg_state.action_enabled = os.environ.get(
                    "PMG_ACTION_ENABLED", "1"
                ) == "1"
                runner._pmg_state.initialize_from_stage(
                    stage,
                    UsdGeom.XformCache(Usd.TimeCode.Default()),
                    UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]),
                    tuple(map(float, episode["start"])),
                    start_yaw=float(episode.get("start_yaw", 0.0)),
                )
                runner._pmg_deferred_observations = []
                # Stage 2 does not instantiate a learned/vector adapter.  Keep
                # the legacy 512-d representation behind an explicit switch so
                # old logging experiments can still be reproduced.
                runner._pmg_adapter512 = None
                runner._pmg_vector512 = []
                if os.environ.get("PMG_ENABLE_VECTOR_ADAPTER", "0") == "1":
                    from pmg_nav.adapter512 import PMGAdapter512

                    runner._pmg_adapter512 = PMGAdapter512()
                    runner._pmg_vector512 = runner._pmg_adapter512.encode(
                        runner._pmg_state.graph,
                        [item.relation for item in (runner._pmg_state.program.constraints if runner._pmg_state.program else ())],
                        runner._pmg_state.candidates,
                    )
                # M1/M2 may expose the bounded PMG context through the direct
                # goal-condition path. Strict M3 is a bypass ablation: it must
                # compute the same PMG payload without changing the released
                # policy's language input, otherwise replay parity is not
                # attributable to action authority.
                if not runner._pmg_state.m3_parity_profile:
                    os.environ["TICVLA_INSTRUCTION"] = runner._pmg_state.context()
            else:
                runner._pmg_state = None
                runner._pmg_adapter512 = None
                runner._pmg_vector512 = []

            behavior_provenance["robot_prim_path"] = runner._spawned_robot_prim_path
            configured_timeout_frames = int(float(episode["timeout"]) * 30)
            timeout_frames = configured_timeout_frames
            if args.smoke_frames:
                timeout_frames = min(timeout_frames, int(args.smoke_frames))
            behavior_provenance.update(
                {
                    "configured_timeout_frames": configured_timeout_frames,
                    "effective_timeout_frames": timeout_frames,
                    "smoke_mode": bool(args.smoke_frames),
                }
            )
            print(
                f"[compat] behavior_backend={args.behavior_backend} "
                f"timeout_frames={timeout_frames} "
                f"script_sha256={behavior_provenance['script_sha256']}",
                flush=True,
            )
            print(
                "[parity] stuck recovery disabled: "
                f"{str(stuck_recovery_disabled_enabled()).lower()} "
                f"strict contract: {str(strict_parity_contract_enabled()).lower()}",
                flush=True,
            )

            started = time.time()
            if args.behavior_backend == "manual_behavior":
                # Kit 107's automatic USD-script lifecycle can tear down while
                # a stage is attaching, leaving an empty action trace.  Remove
                # only this robot's script payload, import the generated file,
                # instantiate it once, and drive every callback explicitly.
                behavior_provenance["robot_usd_scripts_removed"] = (
                    remove_robot_usd_behavior_scripts(
                        stage, runner._spawned_robot_prim_path
                    )
                )
                manual_behavior = load_manual_behavior(
                    behavior_script_path,
                    runner._spawned_robot_prim_path,
                    behavior_provenance,
                )
                print(
                    "[manual_behavior] loaded "
                    f"{behavior_provenance['module_name']}.NovaCarterTICVLA "
                    f"for {runner._spawned_robot_prim_path}",
                    flush=True,
                )
                success = await run_navigation_with_manual_behavior(
                    runner,
                    timeout_frames,
                    manual_behavior,
                    behavior_provenance,
                )
            else:
                attach_robot_usd_behavior_script(
                    stage,
                    runner._spawned_robot_prim_path,
                    behavior_script_path,
                )
                if parity_fixed_horizon_enabled():
                    # The released benchmark loop owns its early-success
                    # break.  Temporarily make the goal probe observational so
                    # a fixed-horizon parity run can continue to the same
                    # frame budget as M3 while retaining the first-success
                    # frame in provenance.  Restore the bound method even if
                    # the upstream loop raises.
                    original_goal_probe = runner._is_goal_reached
                    first_goal_frame = None

                    def _parity_goal_probe(*_args, **_kwargs):
                        nonlocal first_goal_frame
                        reached = bool(original_goal_probe(*_args, **_kwargs))
                        if reached and first_goal_frame is None:
                            first_goal_frame = int(runner._simulation_frame_count)
                            behavior_provenance["first_goal_reached_frame"] = first_goal_frame
                        return False

                    runner._is_goal_reached = _parity_goal_probe
                    try:
                        success = await runner._run_navigation(timeout_frames)
                    finally:
                        runner._is_goal_reached = original_goal_probe
                    success = bool(success or first_goal_frame is not None)
                else:
                    success = await runner._run_navigation(timeout_frames)
            finalize_deferred_m3_pmg(runner, behavior_provenance)
            wall_seconds = time.time() - started
            duration_frames = runner._simulation_frame_count - runner._episode_start_frame
            duration_seconds = duration_frames / 30.0
            navigation_error = runner._get_distance_to_goal()
            collision = runner._physical_collision_detected or runner._collision_detected
            episode_result = {
                "episode": episode_name,
                "episode_seed": episode_seed,
                "scene": episode["scene"],
                "start": episode["start"],
                "goal": episode["goal"],
                "instruction": episode["instruction"],
                "timeout": episode["timeout"],
                "robot_type": episode["robot_type"],
                # Preserve the official benchmark condition separately from
                # the compatibility runtime condition.  Isaac Sim 6 cannot
                # instantiate the released Replicator Agent 0.x API, so the
                # runner intentionally executes with zero dynamic people.
                "official_num_people": int(episode.get("official_num_people", episode.get("num_people", 0))),
                "runtime_num_people": int(episode.get("num_people", 0)),
                "success": bool(success),
                "duration": duration_seconds,
                "duration_frames": duration_frames,
                "parity_fixed_horizon": bool(parity_fixed_horizon_enabled()),
                "deterministic_torch_cuda": bool(parity_deterministic_enabled()),
                "stuck_recovery_disabled": bool(stuck_recovery_disabled_enabled()),
                "strict_parity_contract": bool(strict_parity_contract_enabled()),
                "first_goal_reached_frame": behavior_provenance.get(
                    "first_goal_reached_frame"
                ),
                "wall_seconds": wall_seconds,
                "navigation_error": finite_or_none(navigation_error),
                "success_rate": 1.0 if success else 0.0,
                "collision": 1.0 if collision else 0.0,
                "human_collision": 1.0 if runner._collision_detected else 0.0,
                "physical_collision": 1.0 if runner._physical_collision_detected else 0.0,
                "spl": float(runner._calculate_spl(success)),
                "path_length": float(runner._calculate_path_length()),
                "physical_collision_detected": bool(runner._physical_collision_detected),
                "physical_collision_count": int(runner._max_contact_consecutive_frames),
                "compatibility_runner": "isaac_sim_6_without_ira_0x",
                "behavior_backend": args.behavior_backend,
                "behavior_provenance": behavior_provenance,
                # Backward-compatible alias used by the first manual-driver
                # diagnostic artifacts.
                "behavior_execution": behavior_provenance,
                "diagnostic_contract": diagnostic_contract,
                "action_trace": read_action_trace(trace_path),
            }
            episode_result["artifact_status"], episode_result["artifact_invalid_reasons"] = validate_episode_artifact(episode_result)
            # Keep control-gate telemetry in a stable top-level namespace so
            # matrix aggregation can compare M0/M1 without parsing raw JSONL.
            trace_summary = episode_result["action_trace"]
            episode_result["control_gate"] = {
                key: trace_summary.get(key)
                for key in (
                    "control_policy",
                    "pmg_override_count",
                    "pmg_override_fraction",
                    "pmg_terminal_override_count",
                    "pmg_global_override_count",
                    "pmg_detour_override_count",
                    "policy_action_count",
                    "control_source_counts",
                    "control_scope_counts",
                    "pmg_control_candidate_count",
                    "pmg_control_candidate_first",
                    "pmg_control_candidate_last",
                    "pmg_semantic_candidate_first",
                    "pmg_semantic_candidate_last",
                    "pmg_route_enabled",
                    "pmg_route_valid_count",
                    "pmg_route_output_count",
                    "pmg_route_deferred_output_count",
                    "pmg_route_alignment_hold_count",
                    "pmg_route_alignment_motion_violation_count",
                    "pmg_route_stall_policy_fallback_count",
                    "pmg_route_heading_error_max_abs_rad",
                    "policy_input_checksum",
                    "policy_input_checksum_count",
                    "policy_input_checksum_unique_count",
                    "action_input_checksum",
                    "action_output_checksum",
                    "action_output_count",
                    "parity_greedy_sampling",
                    "parity_greedy_sampling_count",
                    "pmg_payload_checksum",
                    "pmg_payload_checksum_count",
                    "pmg_payload_checksum_unique_count",
                    "simulation_frame_first",
                    "simulation_frame_last",
                    "inference_event_count",
                    "inference_latency_mean_s",
                    "inference_latency_max_s",
                )
            }
            if runner._pmg_state is not None:
                pmg_root = Path(os.environ.get("PMG_METRICS_ROOT", str(Path.cwd() / "pmg_metrics")))
                pmg_output = pmg_root / args.run_id / f"{episode_name}.json"
                runner._pmg_state.save(pmg_output)
                pmg_summary = runner._pmg_state.summary()
                semantic_position = pmg_summary.get("top_candidate_position")
                control_position = pmg_summary.get("control_candidate_position")
                if semantic_position is not None:
                    semantic_position = [float(value) for value in semantic_position]
                    pmg_summary["semantic_goal_pose_error_m"] = math.sqrt(
                        sum((semantic_position[i] - float(episode["goal"][i])) ** 2 for i in range(3))
                    )
                else:
                    pmg_summary["semantic_goal_pose_error_m"] = None
                # The primary goal-pose metric must describe the pose that the
                # controller actually latched/sent. Fall back to semantic top
                # only for profiles that never created a control candidate.
                candidate_position = control_position or semantic_position
                if candidate_position is not None:
                    candidate_position = [float(value) for value in candidate_position]
                    pmg_summary["goal_pose_error_m"] = math.sqrt(
                        sum((candidate_position[i] - float(episode["goal"][i])) ** 2 for i in range(3))
                    )
                else:
                    pmg_summary["goal_pose_error_m"] = None
                # The route planner may select a different feasible candidate
                # from the semantic top-ranked candidate.  Preserve a separate
                # evaluator-only error for that control candidate so reports
                # never conflate graph ranking with the waypoint actually sent
                # to the behavior wrapper.  This calculation occurs after the
                # episode and cannot influence PMG ranking/control.
                route_candidate = pmg_summary.get("route_selected_candidate_position")
                if route_candidate is not None:
                    route_candidate = [float(value) for value in route_candidate]
                    pmg_summary["route_candidate_goal_pose_error_m"] = math.sqrt(
                        sum((route_candidate[i] - float(episode["goal"][i])) ** 2 for i in range(3))
                    )
                else:
                    pmg_summary["route_candidate_goal_pose_error_m"] = None
                pmg_summary["correction_success"] = bool(pmg_summary.get("corrections", 0))
                pmg_summary["verification_revisit_cost"] = float(
                    pmg_summary.get("tool_cost", 0.0)
                ) + float(pmg_summary.get("repeated_exploration_count", 0.0))
                episode_result["pmg"] = pmg_summary
                episode_result["pmg_adapter_dim"] = len(runner._pmg_vector512)
                episode_result["pmg_adapter"] = {
                    "control_adapter": (
                        "PMGGoalAdapter"
                        if pmg_summary.get("goal_adapter_enabled", False)
                        else None
                    ),
                    "direct_goal_condition_used": bool(pmg_summary.get("goal_payload_used", False)) and not bool(pmg_summary.get("goal_adapter_enabled", False)),
                    "control_adapter_consumed": bool(pmg_summary.get("structured_goal_adapter_used", False)),
                    "legacy_vector_adapter": "PMGAdapter512" if runner._pmg_adapter512 is not None else None,
                    "vector_adapter_enabled": runner._pmg_adapter512 is not None,
                    "legacy_vector_consumed": False,
                    "diagnostic_only_payload": bool(
                        pmg_summary.get("diagnostic_only_payload", False)
                    ),
                    "diagnostic_only_payload_calls": int(
                        pmg_summary.get("diagnostic_only_payload_calls", 0)
                    ),
                    "action_authority": "bypassed" if not runner._pmg_state.action_enabled else "enabled",
                }
            try:
                episode_result["peak_gpu_memory_mib"] = (
                    float(torch.cuda.max_memory_allocated() / 2**20)
                    if torch is not None and torch.cuda.is_available()
                    else None
                )
            except Exception:
                episode_result["peak_gpu_memory_mib"] = None
            runner._cleanup_contact_sensor()
            return episode_result

        from omni.kit.async_engine import run_coroutine

        result = wait_for_task(sim_app, run_coroutine(run_episode()))
    except Exception as exc:
        import traceback

        traceback.print_exc()
        result = {
            "episode": episode_name,
            "episode_seed": episode_seed,
            "scene": episode.get("scene"),
            "start": episode.get("start"),
            "goal": episode.get("goal"),
            "instruction": episode.get("instruction", ""),
            "timeout": episode.get("timeout", 0),
            "robot_type": episode.get("robot_type", ""),
            "official_num_people": int(episode.get("official_num_people", episode.get("num_people", 0))),
            "runtime_num_people": int(episode.get("num_people", 0)),
            "success": False,
            "duration": 0.0,
            "duration_frames": 0,
            "navigation_error": None,
            "success_rate": 0.0,
            "collision": 0.0,
            "human_collision": 0.0,
            "physical_collision": 0.0,
            "spl": 0.0,
            "path_length": 0.0,
            "parity_fixed_horizon": bool(parity_fixed_horizon_enabled()),
            "deterministic_torch_cuda": bool(parity_deterministic_enabled()),
            "stuck_recovery_disabled": bool(stuck_recovery_disabled_enabled()),
            "strict_parity_contract": bool(strict_parity_contract_enabled()),
            "physical_collision_detected": False,
            "behavior_backend": args.behavior_backend,
            "behavior_provenance": behavior_provenance,
            "behavior_execution": behavior_provenance,
            "diagnostic_contract": diagnostic_contract,
            "action_trace": read_action_trace(trace_path),
            "peak_gpu_memory_mib": None,
            "error": f"compatibility_runner:{exc}",
            "pmg": {"error": str(exc), "method": str(os.environ.get("PMG_METHOD", "M0"))},
        }
        result["artifact_status"], result["artifact_invalid_reasons"] = validate_episode_artifact(result)
    output_dir = Path.cwd() / "benchmark_results" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.config.stem}_results_latest.json"
    payload = aggregate_result(args.config, result)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"ISAAC6_DYNANAV_RESULT={output_path}", flush=True)
    if result.get("error"):
        raise SystemExit(1)
    # Isaac Sim 6 can block indefinitely in ``SimulationApp.close()`` after a
    # successful headless run (RTX Hydra teardown). The result is already
    # durable at this point; exiting the runner lets the wrapper archive it
    # and prevents one teardown hang from stopping the full matrix.
    os._exit(0)


if __name__ == "__main__":
    main()
