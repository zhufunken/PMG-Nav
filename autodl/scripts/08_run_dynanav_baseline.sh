#!/usr/bin/env bash
set -euo pipefail

ticvla_work_root="${TICVLA_WORK_ROOT:-/root/autodl-tmp/ccfb-ticvla}"
ticvla_repo_dir="${TICVLA_REPO_DIR:-${ticvla_work_root}/src/TIC-VLA}"
isaac_sim_root="${ISAAC_SIM_ROOT:-${ticvla_work_root}/isaacsim}"
isaac_sim_python="${ISAAC_SIM_PYTHON:-${isaac_sim_root}/python.sh}"
config_path="${1:-${ticvla_repo_dir}/DynaNav/configs/benchmark_example.yaml}"
shift || true

for required_path in \
  "${ticvla_repo_dir}/DynaNav/benchmark.py" \
  "${ticvla_repo_dir}/DynaNav/generate_commands.py" \
  "${isaac_sim_python}" \
  "${config_path}" \
  "${TICVLA_BASE_MODEL_PATH:-}" \
  "${TICVLA_CHECKPOINT_PATH:-}"; do
  if [[ -z "${required_path}" || ! -e "${required_path}" ]]; then
    echo "ERROR: required Stage 1 path is missing: ${required_path:-<unset>}" >&2
    exit 2
  fi
done

# Each run gets a private Kit/runtime workspace. This permits bounded method
# sharding without races while keeping PMG metrics in one central directory.
runtime_dir="${ticvla_work_root}/runtime/dynanav/${TICVLA_DYNANAV_RUN_ID:-task7}"
pmg_metrics_root="${ticvla_work_root}/runtime/dynanav/pmg_metrics"
manifest_dir="${ticvla_work_root}/manifests"
output_root="${ticvla_work_root}/outputs/dynanav_closed_loop"
run_id="${TICVLA_DYNANAV_RUN_ID:-task7}"
mkdir -p "${runtime_dir}/behavior" "${manifest_dir}" "${output_root}/${run_id}"
cp -a "${ticvla_repo_dir}/DynaNav/behavior/." "${runtime_dir}/behavior/"

isaac6_compat="${TICVLA_ISAAC6_COMPAT:-0}"
if [[ "${isaac_sim_root##*/}" == isaacsim-6.* ]]; then
  isaac6_compat=1
fi

source_script="${ticvla_repo_dir}/DynaNav/benchmark.py"
headless_script="${runtime_dir}/benchmark_stage1_headless.py"

SOURCE_SCRIPT="${source_script}" HEADLESS_SCRIPT="${headless_script}" env -u PYTHONPATH -u PYTHONHOME -u PYTHONEXE -u LD_PRELOAD python3 - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["SOURCE_SCRIPT"])
target = Path(os.environ["HEADLESS_SCRIPT"])
text = source.read_text()
replacements = {
    'APP_CONFIG = {"renderer": "RayTracedLighting", "headless": False, "width": 640, "height": 480}':
        'APP_CONFIG = {"renderer": "RayTracedLighting", "headless": True, "width": 640, "height": 480}',
    '"omni.anim.people", "omni.isaac.sensor", "isaacsim.replicator.agent.core",':
        '"omni.anim.people", "isaacsim.sensors.physics", "isaacsim.sensors.camera", "isaacsim.replicator.agent.core",',
    '                await self._update_robot_position()\n                self._simulation_frame_count += 1':
        '                await self._update_robot_position()\n                self._simulation_frame_count += 1',
}
for old, new in replacements.items():
    if text.count(old) != 1:
        raise SystemExit(f"ERROR: upstream benchmark no longer matches the locked Stage 1 patch: {old}")
    text = text.replace(old, new)
# Update PMG after Isaac's app update so its USD scan cannot delay the
# behavior callback that advances the learned policy. The candidate is then
# available to the next frame, which is sufficient for bounded recovery.
post_update_marker = '                # advance one frame\n                await app.next_update_async()\n'
if text.count(post_update_marker) != 1:
    raise SystemExit("ERROR: benchmark app-update marker was not found")
pmg_update = (
    '                # Keep a common simulation-frame clock in every method\n'
    '                # trace, including M0 where no PMG state exists.\n'
    '                os.environ["BENCHMARK_SIMULATION_FRAME_COUNT"] = str(self._simulation_frame_count)\n'
    '                # PMG runs after the app update to preserve VLA callback timing.\n'
    '                if getattr(self, "_pmg_state", None) is not None and self._robot_position is not None:\n'
    '                    self._pmg_state.update_from_robot(self._simulation_frame_count, tuple(map(float, self._robot_position)))\n'
    '                    _pmg_diagnostic_only = bool(\n'
    '                        getattr(self._pmg_state, "m3_parity_profile", False)\n'
    '                        and not getattr(self._pmg_state, "action_enabled", True)\n'
    '                    )\n'
    '                    pmg_payload = self._pmg_state.action_payload(\n'
    '                        tuple(map(float, self._robot_position)),\n'
    '                        diagnostic_only=_pmg_diagnostic_only,\n'
    '                    )\n'
    '                    # Clear the previous frame before publishing the new\n'
    '                    # payload.  This prevents a stale waypoint/goal from\n'
    '                    # surviving a transient candidate miss or route failure.\n'
    '                    for _pmg_key in (\n'
    '                        "TICVLA_PMG_GOAL_X", "TICVLA_PMG_GOAL_Y", "TICVLA_PMG_GOAL_Z",\n'
    '                        "TICVLA_PMG_GOAL_DISTANCE", "TICVLA_PMG_GOAL_YAW",\n'
    '                        "TICVLA_PMG_SCORE", "TICVLA_PMG_HARD_VALID", "TICVLA_PMG_STABLE",\n'
    '                        "TICVLA_PMG_WAYPOINT_X", "TICVLA_PMG_WAYPOINT_Y", "TICVLA_PMG_WAYPOINT_Z",\n'
    '                        "TICVLA_PMG_WAYPOINT_INDEX", "TICVLA_PMG_ROUTE_LENGTH",\n'
    '                        "TICVLA_PMG_ROUTE_ENABLED", "TICVLA_PMG_ROUTE_VALID", "TICVLA_PMG_ROUTE_REASON",\n'
    '                        "TICVLA_PMG_SCGP_ENABLED", "TICVLA_PMG_SCGP_ALLOW", "TICVLA_PMG_SCGP_MODE",\n'
    '                        "TICVLA_PMG_SCGP_REASON", "TICVLA_PMG_SCGP_LEASE_UNTIL",\n'
    '                        "TICVLA_PMG_SCGP_STABLE_COUNT", "TICVLA_PMG_SCGP_NO_PROGRESS_TICKS",\n'
    '                        "TICVLA_PMG_DIAGNOSTIC_ONLY",\n'
    '                    ):\n'
    '                        os.environ.pop(_pmg_key, None)\n'
    '                    if pmg_payload is not None and not _pmg_diagnostic_only:\n'
    '                        os.environ.update(pmg_payload)\n'
    '                    else:\n'
    '                        os.environ["TICVLA_PMG_ROUTE_VALID"] = "0"\n'
    '                        os.environ["TICVLA_PMG_ROUTE_ENABLED"] = "0"\n'
    '                        os.environ["TICVLA_PMG_ROUTE_REASON"] = "no_pmg_payload"\n'
    '                        os.environ["TICVLA_PMG_SCORE"] = "0"\n'
    '                        os.environ["TICVLA_PMG_HARD_VALID"] = "0"\n'
    '                        os.environ["TICVLA_PMG_STABLE"] = "0"\n'
    '                        os.environ["TICVLA_PMG_SCGP_ENABLED"] = "0"\n'
    '                        os.environ["TICVLA_PMG_SCGP_ALLOW"] = "1"\n'
    '                        os.environ["TICVLA_PMG_SCGP_MODE"] = "disabled"\n'
    '                        os.environ["TICVLA_PMG_SCGP_REASON"] = "no_pmg_payload"\n'
    '                        if _pmg_diagnostic_only:\n'
    '                            os.environ["TICVLA_PMG_DIAGNOSTIC_ONLY"] = "1"\n'
)
text = text.replace(post_update_marker, post_update_marker + pmg_update)
# Keep the released benchmark instruction authoritative. PMG state is passed
# only through structured recovery fields and the metrics artifact.
text = text.replace(
    '                    if self._simulation_frame_count % self._pmg_state.frame_stride == 0:\n                        os.environ["TICVLA_INSTRUCTION"] = self._pmg_state.context()\n',
    '',
)
target.write_text(text)
PY

diff -u "${source_script}" "${headless_script}" \
  > "${manifest_dir}/task7_headless.patch" || true

if [[ "${isaac6_compat}" == "1" ]]; then
  experience_source="${isaac_sim_root}/apps/isaacsim.exp.base.kit"
  experience_path="${isaac_sim_root}/apps/isaacsim.exp.base.ticvla.kit"
  EXPERIENCE_SOURCE="${experience_source}" EXPERIENCE_PATH="${experience_path}" env -u PYTHONPATH -u PYTHONHOME -u PYTHONEXE -u LD_PRELOAD python3 - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["EXPERIENCE_SOURCE"])
target = Path(os.environ["EXPERIENCE_PATH"])
text = source.read_text(encoding="utf-8")
deprecated_ml_archive = '"omni.isaac.ml_archive" = { order = -100 }\n'
# The first compatibility run creates a cleaned experience file.  Subsequent
# paired runs must be able to regenerate a fresh per-run target even when the
# installed base file has already been cleaned by an earlier setup step.
archive_count = text.count(deprecated_ml_archive)
if archive_count not in (0, 1):
    raise SystemExit("ERROR: Isaac 6 base experience contains an unexpected number of deprecated ML archives")
text = text.replace(deprecated_ml_archive, "")
target.write_text(text, encoding="utf-8")
PY
  diff -u "${experience_source}" "${experience_path}" \
    > "${manifest_dir}/${run_id}_isaac6_experience.patch" || true

  behavior_script="${runtime_dir}/behavior/nova_carter_test_ticvla.py"
  BEHAVIOR_SCRIPT="${behavior_script}" env -u PYTHONPATH -u PYTHONHOME -u PYTHONEXE -u LD_PRELOAD python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["BEHAVIOR_SCRIPT"])
text = path.read_text(encoding="utf-8")
# State JSON is diagnostic telemetry, not a control input. Sampling it at 30
# frames keeps the trajectory auditable while reducing filesystem I/O.
old_log_interval = '        self._log_interval_frames = 10'
if text.count(old_log_interval) != 1:
    raise SystemExit("ERROR: expected behavior telemetry interval was not found")
text = text.replace(old_log_interval, '        self._log_interval_frames = int(os.getenv("TICVLA_LOG_INTERVAL_FRAMES", "30"))')
# The third-person camera is visualization-only; TIC-VLA consumes the front
# camera history above. Keep sparse frames for the artifact archiver.
old_tp_guard = '            if self._image_buffer_dir is not None and self._camera_image_tp is not None:'
new_tp_guard = '''            if (
                self._sensor_frame_count % max(1, int(os.getenv("TICVLA_TP_SAVE_EVERY_N", "1"))) == 0
                and self._image_buffer_dir is not None
                and self._camera_image_tp is not None
            ):'''
if text.count(old_tp_guard) != 1:
    raise SystemExit("ERROR: expected third-person image guard was not found")
text = text.replace(old_tp_guard, new_tp_guard)
# Camera render products are separate from Kit's viewport resolution.  Keep
# the released 1920x1080 behavior by default, while allowing paired resource
# gates to lower both methods' camera footprint without changing PMG logic.
old_camera_resolution = 'resolution=(1920, 1080)'
new_camera_resolution = 'resolution=(int(os.getenv("TICVLA_CAMERA_WIDTH", "1920")), int(os.getenv("TICVLA_CAMERA_HEIGHT", "1080")))'
if text.count(old_camera_resolution) != 2:
    raise SystemExit("ERROR: expected front/third-person camera resolutions were not found")
text = text.replace(old_camera_resolution, new_camera_resolution)
# Isaac Sim 6 compatibility images used in this project still provide
# ``omni.kit.scripting``.  The newer ``omni.behavior.scripting.core`` module
# is not installed in the deployed 107.3.1 extension set and causes Kit to
# block while resolving a missing dependency.  Keep the upstream import so
# the behavior attaches through the extension that is actually available.
if text.count("from omni.kit.scripting import BehaviorScript") != 1:
    raise SystemExit("ERROR: expected omni.kit.scripting import was not found")

old_people = "from omni.anim.people.settings import PeopleSettings, AgentEvent"
new_people = '''try:
    from omni.anim.people.settings import PeopleSettings, AgentEvent
except ImportError:
    class PeopleSettings:
        NAVMESH_ENABLED = "/exts/omni.anim.people/navigation_settings/navmesh_enabled"
        DYNAMIC_AVOIDANCE_ENABLED = "/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled"

    class AgentEvent:
        AgentRegistered = "omni.anim.people/REGISTER_AGENT"'''
if text.count(old_people) != 1:
    raise SystemExit("ERROR: expected People import was not found")
text = text.replace(old_people, new_people)

old_navigation = "from robot_navigation_manager import RobotNavigationManager"
new_navigation = '''runtime_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if runtime_root not in sys.path:
    sys.path.insert(0, runtime_root)
# benchmark imports can cache the upstream module before this behavior script
# is evaluated; load the Isaac 6 shim by path and replace that cache explicitly.
import importlib.util
shim_path = os.path.join(runtime_root, "robot_navigation_manager.py")
shim_spec = importlib.util.spec_from_file_location("robot_navigation_manager", shim_path)
if shim_spec is None or shim_spec.loader is None:
    raise ImportError(f"Isaac 6 navigation shim is missing: {shim_path}")
shim_module = importlib.util.module_from_spec(shim_spec)
sys.modules["robot_navigation_manager"] = shim_module
shim_spec.loader.exec_module(shim_module)
from robot_navigation_manager import RobotNavigationManager'''
if text.count(old_navigation) != 1:
    raise SystemExit("ERROR: expected robot navigation import was not found")
text = text.replace(old_navigation, new_navigation)

old_class = '''class NovaCarterTICVLA(BehaviorScript):
    """TIC-VLA-based autonomous navigation behavior for Nova Carter robot with async inference."""'''
new_class = old_class + '''

    def get_agent_name(self):
        return str(self.prim_path).rstrip("/").rsplit("/", 1)[-1]

    def _stage2_load_policy_prefix(self):
        """Load a frozen M0 command prefix for a causal paired M1 run.

        The prefix is an evaluation-only bridge: before PMG obtains terminal
        authority, M1 replays the exact command emitted by the paired M0 at
        the same simulation frame.  This removes independent TIC-VLA/render
        scheduling from the counterfactual while leaving the PMG hand-off
        itself live and auditable.
        """
        self._stage2_policy_prefix_by_frame = {}
        self._stage2_policy_prefix_path = os.getenv("PMG_POLICY_PREFIX_TRACE", "").strip()
        self._stage2_policy_prefix_enabled = bool(
            self._stage2_policy_prefix_path
            and os.getenv("STAGE2_PAIRED_ACTION_PREFIX", "0").lower()
            in {"1", "true", "yes", "on"}
        )
        self._stage2_policy_prefix_load_error = None
        if not self._stage2_policy_prefix_enabled:
            return
        try:
            with open(self._stage2_policy_prefix_path, "r", encoding="utf-8") as _prefix_file:
                for _prefix_line in _prefix_file:
                    try:
                        _prefix_row = __import__("json").loads(_prefix_line)
                    except Exception:
                        continue
                    if _prefix_row.get("kind") != "output":
                        continue
                    _prefix_frame = _prefix_row.get("simulation_frame")
                    _prefix_v = _prefix_row.get("v_cmd")
                    _prefix_w = _prefix_row.get("w_cmd")
                    if _prefix_frame is None or _prefix_v is None or _prefix_w is None:
                        continue
                    try:
                        _prefix_frame = int(_prefix_frame)
                        _prefix_v = float(_prefix_v)
                        _prefix_w = float(_prefix_w)
                    except (TypeError, ValueError):
                        continue
                    if not (__import__("math").isfinite(_prefix_v) and __import__("math").isfinite(_prefix_w)):
                        continue
                    self._stage2_policy_prefix_by_frame[_prefix_frame] = (_prefix_v, _prefix_w)
        except OSError as _prefix_error:
            self._stage2_policy_prefix_load_error = str(_prefix_error)
        if not self._stage2_policy_prefix_by_frame:
            raise RuntimeError(
                "STAGE2_PAIRED_ACTION_PREFIX requested but M0 action prefix "
                f"could not be loaded: {self._stage2_policy_prefix_path}"
            )
        print(
            "[parity] frozen M0 action prefix loaded "
            f"frames={len(self._stage2_policy_prefix_by_frame)} "
            f"path={self._stage2_policy_prefix_path}",
            flush=True,
        )

    def _pmg_refresh_control_gate(self):
        """Resolve the fair global-policy/local-terminal PMG hand-off.

        The released TIC-VLA policy owns the route by default.  PMG can only
        take over inside a hysteretic terminal radius, unless an explicitly
        named ``global`` policy is requested for an oracle upper-bound run.
        All inputs here are PMG candidate fields and measured robot pose; no
        evaluator goal, success flag, or evaluator trajectory is consulted.
        """
        method = str(getattr(self, "_pmg_method", "M0") or "M0").upper()
        variant = os.getenv("STAGE2_M1_VARIANT", "").lower()
        raw_policy = os.getenv("PMG_CONTROL_POLICY", os.getenv("PMG_CONTROL_MODE", ""))
        if not raw_policy.strip():
            raw_policy = (
                "terminal"
                if method == "M1" and variant == "simple_oracle"
                else ("recovery" if method != "M0" else "policy_only")
            )
        aliases = {
            "policy": "policy_only", "policy_only": "policy_only", "ticvla": "policy_only", "off": "policy_only", "none": "policy_only",
            "terminal": "terminal", "local": "terminal", "local_terminal": "terminal", "fair": "terminal",
            "recovery": "recovery", "terminal_recovery": "recovery",
            "global": "global", "always": "global", "oracle": "global", "global_oracle": "global",
        }
        default_policy = "terminal" if method == "M1" and variant == "simple_oracle" else ("recovery" if method != "M0" else "policy_only")
        self._pmg_control_policy = aliases.get(raw_policy.strip().lower(), default_policy)
        try:
            enter_distance = max(0.0, float(os.getenv("PMG_TERMINAL_ENTER_DISTANCE", "4.0")))
        except (TypeError, ValueError):
            enter_distance = 4.0
        try:
            exit_distance = max(
                enter_distance,
                float(os.getenv("PMG_TERMINAL_EXIT_DISTANCE", str(enter_distance + 1.0))),
            )
        except (TypeError, ValueError):
            exit_distance = enter_distance + 1.0
        self._pmg_terminal_enter_distance = enter_distance
        self._pmg_terminal_exit_distance = exit_distance
        simple_oracle = method == "M1" and variant == "simple_oracle"
        try:
            score_floor = float(
                os.getenv(
                    "PMG_SIMPLE_ORACLE_MIN_SCORE" if simple_oracle else "PMG_RECOVERY_MIN_SCORE",
                    "0.0" if simple_oracle else "0.92",
                )
            )
        except (TypeError, ValueError):
            score_floor = 0.0 if simple_oracle else 0.92
        candidate_valid = bool(
            method != "M0"
            and os.getenv("PMG_ACTION_ENABLED", "1") == "1"
            and self._pmg_goal_position is not None
            and self._pmg_hard_valid
            and (simple_oracle or self._pmg_stable)
            and self._pmg_score >= score_floor
        )
        # When explicitly enabled, SCGP is the additional authority gate. A
        # missing/denied lease must return control to TIC-VLA; the disabled
        # path leaves historical behavior unchanged.
        if bool(getattr(self, "_pmg_scgp_enabled", False)) and not bool(
            getattr(self, "_pmg_scgp_allow", False)
        ):
            candidate_valid = False
        # Keep the terminal hand-off radius tied to the eventual semantic
        # candidate, not to a first route waypoint.  A waypoint can be close
        # to the start simply because it is the first A* cell; using it for
        # gating would silently restore a global PMG takeover.  Once the
        # semantic candidate is local, the control branch below still gives
        # the route waypoint priority over the final candidate.
        control_position = self._pmg_goal_position
        if (
            candidate_valid
            and bool(getattr(self, "_pmg_route_valid", False))
            and getattr(self, "_pmg_waypoint_position", None) is not None
        ):
            control_position = self._pmg_waypoint_position
        self._pmg_control_position = control_position
        semantic_distance = float("inf")
        candidate_distance = float("inf")
        if candidate_valid and self._robot_position is not None:
            try:
                semantic_delta = np.asarray(self._pmg_goal_position[:2], dtype=np.float64) - np.asarray(self._robot_position[:2], dtype=np.float64)
                semantic_distance = float(np.linalg.norm(semantic_delta))
                # candidate_distance is the gate distance and intentionally
                # remains semantic-goal distance.  The separate control
                # position is only consumed after the gate opens.
                candidate_distance = semantic_distance
            except (TypeError, ValueError, IndexError):
                candidate_valid = False
        if not np.isfinite(candidate_distance) or candidate_distance < 0.0:
            candidate_valid = False
        previous_active = bool(getattr(self, "_pmg_terminal_active", False))
        checkpoint_released = bool(
            simple_oracle
            and os.getenv("PMG_GOAL_ROLE", "constraint_checkpoint").lower()
            == "constraint_checkpoint"
            and os.getenv("PMG_SIMPLE_ORACLE_RELEASE_ON_ARRIVAL", "0").lower()
            in {"1", "true", "yes", "on"}
            and getattr(self, "_pmg_checkpoint_released", False)
        )
        if checkpoint_released:
            # A semantic PMG candidate can be an intermediate checkpoint. Once
            # reached, release authority so the official TIC-VLA evaluator
            # route can continue to its final goal.
            terminal_active = False
            gate_reason = "checkpoint_released"
        elif not candidate_valid:
            terminal_active = False
            gate_reason = "candidate_invalid"
        elif previous_active:
            terminal_active = candidate_distance <= exit_distance
            gate_reason = "terminal_active" if terminal_active else "outside_terminal_exit_radius"
        else:
            terminal_active = candidate_distance <= enter_distance
            gate_reason = "terminal_enter" if terminal_active else "outside_terminal_radius"
        self._pmg_terminal_active = bool(terminal_active)
        self._pmg_terminal_eligible = bool(candidate_valid and terminal_active)
        self._pmg_candidate_valid = bool(candidate_valid)
        self._pmg_candidate_distance_actual = candidate_distance if candidate_valid else None
        self._pmg_semantic_candidate_distance_actual = semantic_distance if candidate_valid else None
        self._pmg_global_override = bool(
            candidate_valid and self._pmg_control_policy == "global" and not checkpoint_released
        )
        self._pmg_gate_reason = (
            f"scgp:{getattr(self, '_pmg_scgp_reason', 'deny')}"
            if bool(getattr(self, "_pmg_scgp_enabled", False))
            and not bool(getattr(self, "_pmg_scgp_allow", False))
            else gate_reason
        )
'''
if text.count(old_class) != 1:
    raise SystemExit("ERROR: expected Nova Carter behavior class declaration was not found")
text = text.replace(old_class, new_class)

old_pmg_init = '''        self._goal_position = None
        self._instruction = ""'''
new_pmg_init = '''        self._goal_position = None
        self._pmg_goal_position = None
        self._pmg_waypoint_position = None
        self._pmg_route_enabled = False
        self._pmg_route_valid = False
        self._pmg_route_reason = "not_provided"
        self._pmg_waypoint_index = -1
        self._pmg_route_length = float("inf")
        self._pmg_scgp_enabled = False
        self._pmg_scgp_allow = True
        self._pmg_scgp_mode = "disabled"
        self._pmg_scgp_reason = "disabled"
        self._pmg_scgp_lease_until = -1
        self._pmg_scgp_stable_count = 0
        self._pmg_scgp_no_progress_ticks = 0
        self._pmg_score = 0.0
        self._pmg_hard_valid = False
        self._pmg_stable = False
        self._pmg_goal_distance = float("inf")
        self._pmg_method = "M0"
        # PMG direct pursuit is allowed to yield to the learned policy when
        # the robot is physically blocked.  This is a controller-side
        # recovery signal based only on observed motion, never evaluator goal.
        self._pmg_stall_frames = 0
        self._pmg_detour_frames = 0
        self._pmg_detour_side = 1.0
        self._pmg_direct_stall_key = None
        self._pmg_direct_detour_attempted = False
        self._pmg_direct_stall_latched = False
        # Route progress watchdog.  A stationary angular controller can keep
        # reporting valid alignment while the measured distance to the active
        # waypoint never improves.  Track geometric progress independently of
        # commanded speed so this failure mode releases control deterministically.
        self._pmg_progress_waypoint_key = None
        self._pmg_progress_best_distance = float("inf")
        self._pmg_progress_stall_inferences = 0
        self._pmg_progress_release_count = 0
        self._pmg_progress_release_key = None
        self._pmg_progress_release_best_distance = float("inf")
        self._pmg_progress_release_attempted = False
        self._pmg_route_stall_latched = False
        # Fair-control gate state.  TIC-VLA owns the global route; PMG only
        # receives a local terminal hand-off unless an explicit global oracle
        # policy is requested.  Counters are emitted in action telemetry.
        self._pmg_control_policy = "policy_only"
        self._pmg_terminal_active = False
        self._pmg_checkpoint_released = False
        self._pmg_terminal_eligible = False
        self._pmg_candidate_valid = False
        self._pmg_candidate_distance_actual = None
        self._pmg_semantic_candidate_distance_actual = None
        self._pmg_control_position = None
        self._pmg_terminal_enter_distance = 4.0
        self._pmg_terminal_exit_distance = 5.0
        self._pmg_global_override = False
        self._pmg_gate_reason = "initialization"
        self._pmg_action_ticks = 0
        self._pmg_override_ticks = 0
        self._pmg_terminal_override_ticks = 0
        self._pmg_global_override_ticks = 0
        self._pmg_detour_override_ticks = 0
        self._pmg_policy_ticks = 0
        self._pmg_route_aligning = False
        self._pmg_last_control_source = "policy"
        self._pmg_last_override = False
        self._instruction = ""'''
if text.count(old_pmg_init) != 2:
    raise SystemExit("ERROR: expected behavior goal initialization block was not found")
text = text.replace(old_pmg_init, new_pmg_init)

# Load the paired M0 command prefix after the common behavior state exists.
# The call is harmless for ordinary Stage 1/Stage 2 runs because the switch is
# opt-in, and it fails closed when a requested prefix is absent.
prefix_init_marker = '        self._instruction = ""'
prefix_init_replacement = prefix_init_marker + '''
        self._stage2_load_policy_prefix()'''
if text.count(prefix_init_marker) != 2:
    raise SystemExit("ERROR: expected behavior instruction initialization blocks were not found")
text = text.replace(prefix_init_marker, prefix_init_replacement)

# A deliberate PMG terminal controller must be allowed to rotate in place
# while it acquires the goal bearing.  The upstream generic stuck detector
# only watches translational displacement and can interpret a valid terminal
# turn as a deadlock.  Crucially, exempt it only while the fair gate has
# actually entered the terminal scope (or for an explicit global oracle run);
# from the start of an episode the learned TIC-VLA policy keeps its normal
# safety behavior.
old_stuck_gate = '        if self._first_inference_complete and not self._is_backing_up and not self._waiting_for_inference_after_backup:'
new_stuck_gate = '''        if (
            self._first_inference_complete
            and not self._is_backing_up
            and not self._waiting_for_inference_after_backup
            # Strict M0/M3 replay compares PMG decision flow, not the
            # asynchronous legacy backup heuristic. That heuristic depends
            # on callback timing and can trigger at different frames in
            # otherwise paired runs, changing the action horizon.
            and os.getenv("STAGE2_DISABLE_STUCK_RECOVERY", "0").lower()
            not in {"1", "true", "yes", "on"}
            and not (
                os.getenv("PMG_METHOD", "M0").upper() == "M1"
                and os.getenv("STAGE2_M1_VARIANT", "").lower() == "simple_oracle"
                and os.getenv("PMG_ACTION_ENABLED", "1") == "1"
                and (
                    bool(getattr(self, "_pmg_terminal_active", False))
                    or bool(getattr(self, "_pmg_global_override", False))
                )
            )
        ):'''
if text.count(old_stuck_gate) != 1:
    raise SystemExit("ERROR: expected stuck-detection gate was not found")
text = text.replace(old_stuck_gate, new_stuck_gate)

# A strict paired replay can start with different first RGB availability when
# the PMG state is initialized during stage setup.  An explicit minimum action
# frame keeps every method stationary until the same first query frame, then
# trims the image history to the newest frame so the policy sees the same
# warm-up horizon.  The switch is opt-in and has no effect on ordinary runs.
parity_min_action_gate = '''            # Decide whether to run (or schedule) TIC-VLA inference
            # Skip inference during backup recovery, but allow it when waiting for inference after backup
            have_images = len(self._image_history) > 0 and self._model_loaded and (self._ticvla_model is not None) and not self._is_backing_up'''
parity_min_action_gate_replacement = '''            # Decide whether to run (or schedule) TIC-VLA inference
            # Skip inference during backup recovery, but allow it when waiting for inference after backup
            # Strict paired runs may hold the first action until every method
            # reaches the same frame.  Trim
            # stale warm-up images at the release frame so both policies see
            # the same latest observation rather than hidden pre-gate history.
            _parity_min_action_frame = 0
            try:
                _parity_min_action_frame = int(os.getenv("PMG_PARITY_MIN_ACTION_FRAME", "0"))
            except (TypeError, ValueError):
                _parity_min_action_frame = 0
            _parity_mode = str(
                os.getenv("STAGE2_DIAGNOSTIC_MODE")
                or os.getenv("PMG_METHOD", "M0")
            ).upper()
            _parity_fixed_horizon = any(
                os.getenv(_parity_flag, "0").lower()
                in {"1", "true", "yes", "on"}
                for _parity_flag in (
                    "STAGE2_PARITY_FIXED_HORIZON",
                    "PMG_PARITY_FIXED_HORIZON",
                )
            )
            _parity_gate_active = (
                _parity_min_action_frame > 0
                and _parity_mode in {"M0", "M1", "M2", "M3"}
                and _parity_fixed_horizon
            )
            if _parity_gate_active and self._frame_count < _parity_min_action_frame:
                have_images = False
            else:
                if (
                    _parity_gate_active
                    and self._frame_count >= _parity_min_action_frame
                    and not getattr(
                        self, "_stage2_parity_min_action_released", False
                    )
                ):
                    _stale_warmup_paths = self._image_history[:-1]
                    self._image_history = self._image_history[-1:]
                    for _stale_warmup_path in _stale_warmup_paths:
                        try:
                            if os.path.exists(_stale_warmup_path):
                                os.remove(_stale_warmup_path)
                        except OSError:
                            pass
                    self._stage2_parity_min_action_released = True
                have_images = len(self._image_history) > 0 and self._model_loaded and (self._ticvla_model is not None) and not self._is_backing_up'''
if text.count(parity_min_action_gate) != 1:
    raise SystemExit("ERROR: expected inference scheduling block was not found")
text = text.replace(parity_min_action_gate, parity_min_action_gate_replacement)

inference_call = "        with torch.no_grad():\n            response, waypoints, vlm_generation_start_step, kv_cache_available, vlm_generation_start_pose = self._ticvla_model.predict_async("
if text.count(inference_call) != 1:
    raise SystemExit("ERROR: expected TIC-VLA inference call was not found")
text = text.replace(
    inference_call,
    '''        # Diagnostic input trace: this is the structured state visible to
        # the released query.  The 512-d PMG vector is deliberately marked
        # unconsumed because no projector is wired into TIC-VLA in Stage 2.
        # This function runs in a separate inference-task scope from the
        # scheduling block above, so parity gate metadata is recomputed here
        # instead of relying on that block's local variables.
        try:
            _trace_parity_min_action_frame = int(
                os.getenv("PMG_PARITY_MIN_ACTION_FRAME", "0")
            )
        except (TypeError, ValueError):
            _trace_parity_min_action_frame = 0
        _trace_parity_mode = str(
            os.getenv("STAGE2_DIAGNOSTIC_MODE")
            or os.getenv("PMG_METHOD", "M0")
        ).upper()
        _trace_parity_gate_active = bool(
            _trace_parity_min_action_frame > 0
            and _trace_parity_mode in {"M0", "M1", "M2", "M3"}
            and any(
                os.getenv(_trace_parity_flag, "0").lower()
                in {"1", "true", "yes", "on"}
                for _trace_parity_flag in (
                    "STAGE2_PARITY_FIXED_HORIZON",
                    "PMG_PARITY_FIXED_HORIZON",
                )
            )
        )
        diagnostic_input = {
            "kind": "input",
            "mode": os.getenv("STAGE2_DIAGNOSTIC_MODE", os.getenv("PMG_METHOD", "M0")),
            "instruction": self._instruction,
            "pmg_action_enabled": os.getenv("PMG_ACTION_ENABLED", "1") == "1",
            "pmg_goal": [os.getenv("TICVLA_PMG_GOAL_X"), os.getenv("TICVLA_PMG_GOAL_Y"), os.getenv("TICVLA_PMG_GOAL_Z")],
            "pmg_waypoint": [os.getenv("TICVLA_PMG_WAYPOINT_X"), os.getenv("TICVLA_PMG_WAYPOINT_Y"), os.getenv("TICVLA_PMG_WAYPOINT_Z")],
            "pmg_route_valid": os.getenv("TICVLA_PMG_ROUTE_VALID", "0") == "1",
            "pmg_route_enabled": os.getenv("TICVLA_PMG_ROUTE_ENABLED", "0") == "1",
            "pmg_route_reason": os.getenv("TICVLA_PMG_ROUTE_REASON"),
            "pmg_control_policy": os.getenv("PMG_CONTROL_POLICY", os.getenv("PMG_CONTROL_MODE", "")),
            "pmg_terminal_enter_distance": os.getenv("PMG_TERMINAL_ENTER_DISTANCE", "4.0"),
            "pmg_terminal_exit_distance": os.getenv("PMG_TERMINAL_EXIT_DISTANCE", "5.0"),
            "pmg_vector_consumed": False,
        }
        diagnostic_input["checksum"] = __import__("hashlib").sha256(
            __import__("json").dumps(diagnostic_input, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        # Keep a second checksum for *actual TIC-VLA query inputs*.  The
        # historical ``checksum`` above intentionally includes PMG metadata
        # and therefore must differ between M0 and PMG methods.  For parity,
        # image paths are canonicalized to remove each run's private output
        # root; otherwise identical paired frames would hash differently just
        # because their temporary directories differ.
        _trace_paths = locals().get("image_paths")
        if _trace_paths is None:
            _trace_paths = locals().get("sampled_paths")
        if _trace_paths is None:
            _trace_paths = []
        def _canonical_trace_image_ref(value):
            raw = str(value).replace("\\\\", "/")
            marker = "/images/"
            if marker in raw:
                return raw.split(marker, 1)[1]
            return raw.rsplit("/", 1)[-1]
        _trace_image_refs = [_canonical_trace_image_ref(path) for path in _trace_paths]
        # A basename/reference match is not enough to establish paired
        # observations: two Isaac processes can write the same frame name
        # while rendering different pixels.  Content hashing is opt-in for
        # the short observation-parity gate because it adds filesystem I/O to
        # every query; normal Stage 2 runs retain the lightweight path trace.
        _trace_content_hash_enabled = any(
            os.getenv(_trace_hash_flag, "0").strip().lower()
            in {"1", "true", "yes", "on"}
            for _trace_hash_flag in ("PMG_TRACE_CONTENT_HASH", "STAGE2_TRACE_CONTENT_HASH")
        )
        _trace_image_sha256s = [None for _ in _trace_paths]
        if _trace_content_hash_enabled:
            _trace_image_digest_cache = getattr(
                self, "_stage2_trace_image_digest_cache", None
            )
            if _trace_image_digest_cache is None:
                _trace_image_digest_cache = {}
                self._stage2_trace_image_digest_cache = _trace_image_digest_cache
            def _trace_image_sha256(value):
                try:
                    resolved = os.path.realpath(str(value))
                    cached = _trace_image_digest_cache.get(resolved)
                    if cached is not None:
                        return cached
                    digest = __import__("hashlib").sha256()
                    with open(resolved, "rb") as _trace_image_file:
                        for _trace_chunk in iter(
                            lambda: _trace_image_file.read(1024 * 1024), b""
                        ):
                            digest.update(_trace_chunk)
                    cached = digest.hexdigest()
                    _trace_image_digest_cache[resolved] = cached
                    return cached
                except (OSError, TypeError, ValueError):
                    return None
            _trace_image_sha256s = [_trace_image_sha256(path) for path in _trace_paths]
        def _trace_float_list(value):
            try:
                return [float(item) for item in value]
            except (TypeError, ValueError):
                return None
        _trace_robot_state = _trace_float_list(robot_state_tuple)
        _trace_robot_pose = None
        if isinstance(robot_pose, dict):
            _trace_robot_pose = {
                "position": _trace_float_list(robot_pose.get("position")),
                "quaternion": _trace_float_list(robot_pose.get("quaternion")),
            }
        _trace_policy_input = {
            "instruction": self._instruction,
            "image_refs": _trace_image_refs,
            "image_sha256": _trace_image_sha256s,
            "robot_state": _trace_robot_state,
            "robot_pose": _trace_robot_pose,
            "delay_time": float(delay_time) if delay_time is not None else None,
            "robot_type": str(robot_type),
            "inference_step": int(current_step) if current_step is not None else None,
        }
        diagnostic_input["policy_input_refs"] = _trace_image_refs
        diagnostic_input["policy_input_checksum"] = __import__("hashlib").sha256(
            __import__("json").dumps(
                _trace_policy_input,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        # Keep the checksum preimage in parity traces.  A digest alone cannot
        # explain a same-frame mismatch (for example a different normalized
        # robot_type or delay_time); this bounded payload is metadata only and
        # is never passed to TIC-VLA.
        diagnostic_input["policy_input_payload"] = _trace_policy_input
        diagnostic_input["trace_content_hash_enabled"] = bool(_trace_content_hash_enabled)
        # The PMG payload hash is deliberately separate from the policy hash.
        # It proves which structured fields were available at this query while
        # making clear that those fields are not part of M0/M3 model parity.
        diagnostic_input["parity_min_action_frame"] = int(
            _trace_parity_min_action_frame
        )
        diagnostic_input["parity_min_action_gate_active"] = bool(
            _trace_parity_gate_active
        )
        diagnostic_input["parity_min_action_released"] = bool(
            getattr(self, "_stage2_parity_min_action_released", False)
        )
        _trace_pmg_keys = (
            "TICVLA_PMG_GOAL_X", "TICVLA_PMG_GOAL_Y", "TICVLA_PMG_GOAL_Z",
            "TICVLA_PMG_GOAL_DISTANCE", "TICVLA_PMG_GOAL_YAW", "TICVLA_PMG_SCORE",
            "TICVLA_PMG_HARD_VALID", "TICVLA_PMG_STABLE",
            "TICVLA_PMG_WAYPOINT_X", "TICVLA_PMG_WAYPOINT_Y", "TICVLA_PMG_WAYPOINT_Z",
            "TICVLA_PMG_WAYPOINT_INDEX", "TICVLA_PMG_ROUTE_LENGTH",
            "TICVLA_PMG_ROUTE_ENABLED", "TICVLA_PMG_ROUTE_VALID", "TICVLA_PMG_ROUTE_REASON",
            "TICVLA_PMG_SCGP_ENABLED", "TICVLA_PMG_SCGP_ALLOW", "TICVLA_PMG_SCGP_MODE",
            "TICVLA_PMG_SCGP_REASON", "TICVLA_PMG_SCGP_LEASE_UNTIL",
            "TICVLA_PMG_SCGP_STABLE_COUNT", "TICVLA_PMG_SCGP_NO_PROGRESS_TICKS",
            "TICVLA_PMG_DIAGNOSTIC_ONLY",
        )
        _trace_pmg_payload = {key: os.getenv(key) for key in _trace_pmg_keys}
        diagnostic_input["pmg_payload_checksum"] = __import__("hashlib").sha256(
            __import__("json").dumps(
                _trace_pmg_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        diagnostic_input["pmg_payload_present"] = any(
            value is not None for value in _trace_pmg_payload.values()
        )
        # Timing fields are appended after the stable checksum so paired
        # input-state checksums remain comparable across runs.
        diagnostic_input["simulation_frame"] = int(os.getenv("BENCHMARK_SIMULATION_FRAME_COUNT", "-1"))
        diagnostic_input["monotonic_seconds"] = time.perf_counter()
        diagnostic_input["inference_step"] = int(current_step) if current_step is not None else None
        diagnostic_trace_path = os.getenv("PMG_ACTION_TRACE_PATH")
        # TIC-VLA's released ``predict_async`` path deliberately samples from
        # InternVL (do_sample=True, temperature/top-k/top-p).  That stochastic
        # generation made otherwise identical M0/M3 queries diverge at the
        # first action, so a parity run may opt into a wrapper around the
        # shared ``generate_and_extract_kv_cache`` helper.  The wrapper is
        # local to this generated behavior instance and leaves upstream
        # TIC-VLA untouched; ordinary Stage 2 runs keep the released sampler.
        _stage2_parity_greedy = any(
            os.getenv(_stage2_flag, "0").strip().lower()
            in {"1", "true", "yes", "on"}
            for _stage2_flag in (
                "STAGE2_PARITY_DETERMINISTIC",
                "PMG_PARITY_DETERMINISTIC",
                "STAGE2_PARITY_DETERMINISTIC_ACTIVE",
            )
        )
        if _stage2_parity_greedy and not getattr(self, "_stage2_parity_greedy_wrapped", False):
            _stage2_model = getattr(self, "_ticvla_model", None)
            _stage2_original_generate = getattr(
                _stage2_model, "generate_and_extract_kv_cache", None
            )
            if _stage2_model is not None and callable(_stage2_original_generate):
                def _stage2_greedy_generate(*_args, **_kwargs):
                    # The TIC-VLA helper receives generation_config as its
                    # third positional argument in released checkpoints, but
                    # accept a keyword form too for compatibility snapshots.
                    args_list = list(_args)
                    config = _kwargs.get("generation_config")
                    config_index = None
                    if config is None:
                        for _index, _value in enumerate(args_list):
                            # Released snapshots pass a plain dict, while
                            # newer Transformers paths may pass a
                            # GenerationConfig object.  Restrict discovery
                            # to the generation-config slot/signature so a
                            # model or image argument with a coincidental
                            # attribute is never rewritten.
                            _looks_like_generation_config = bool(
                                _index >= 2
                                and isinstance(_value, dict)
                                and {
                                    "do_sample", "temperature", "top_p", "top_k"
                                } & set(_value)
                            ) or bool(
                                _index >= 2
                                and any(
                                    hasattr(_value, _stage2_key)
                                    for _stage2_key in (
                                        "do_sample", "temperature", "top_p", "top_k"
                                    )
                                )
                            )
                            if _looks_like_generation_config:
                                config = _value
                                config_index = _index
                                break
                    if config is None and len(args_list) > 2:
                        config_index = 2
                        config = args_list[config_index]
                    # Preserve a GenerationConfig-like object's type when
                    # possible: TIC-VLA helper implementations differ in
                    # whether they access mapping keys or attributes.  A
                    # deepcopy prevents the parity wrapper from mutating the
                    # released sampler shared by later calls.  Fall back to
                    # a plain dict only for immutable/custom objects that
                    # cannot be copied or have their fields updated.
                    _greedy_config = None
                    if config is not None and not isinstance(config, dict):
                        try:
                            _greedy_config = __import__("copy").deepcopy(config)
                        except Exception:
                            try:
                                _greedy_config = __import__("copy").copy(config)
                            except Exception:
                                _greedy_config = None
                        if _greedy_config is not None and hasattr(_greedy_config, "do_sample"):
                            try:
                                _greedy_config.do_sample = False
                                for _key in ("temperature", "top_p", "top_k"):
                                    if hasattr(_greedy_config, _key):
                                        setattr(_greedy_config, _key, None)
                            except Exception:
                                _greedy_config = None
                    if _greedy_config is None:
                        if hasattr(config, "to_dict") and callable(config.to_dict):
                            _greedy_config = dict(config.to_dict())
                        else:
                            _greedy_config = dict(config or {})
                        _greedy_config["do_sample"] = False
                        for _key in ("temperature", "top_p", "top_k"):
                            _greedy_config.pop(_key, None)
                    if "generation_config" in _kwargs:
                        _kwargs["generation_config"] = _greedy_config
                    elif config_index is not None:
                        args_list[config_index] = _greedy_config
                    else:
                        _kwargs["generation_config"] = _greedy_config
                    return _stage2_original_generate(*tuple(args_list), **_kwargs)

                _stage2_model.generate_and_extract_kv_cache = _stage2_greedy_generate
                self._stage2_parity_greedy_wrapped = True
                self._stage2_parity_greedy_sampling = True
                print(
                    "[parity] TIC-VLA generation wrapper enabled: do_sample=False",
                    flush=True,
                )
            else:
                self._stage2_parity_greedy_sampling = False
                # A parity run that silently falls back to do_sample=True is
                # not a valid paired experiment.  Fail before issuing the
                # first action so the caller cannot mistake a stochastic run
                # for a passed parity gate.
                raise RuntimeError(
                    "STAGE2_PARITY_DETERMINISTIC requested but TIC-VLA "
                    "generate_and_extract_kv_cache is unavailable"
                )
        diagnostic_input["parity_greedy_sampling"] = bool(
            getattr(self, "_stage2_parity_greedy_sampling", False)
        )
        if diagnostic_trace_path:
            with open(diagnostic_trace_path, "a", encoding="utf-8") as diagnostic_trace:
                diagnostic_trace.write(__import__("json").dumps(diagnostic_input, sort_keys=True) + "\\n")
        inference_started = time.perf_counter()
''' + inference_call,
)
inference_end = '''                robot_type=robot_type
            )
        
        # Check AGAIN if backup started during model inference (model call takes time)'''
inference_end_with_latency = '''                robot_type=robot_type
            )
        inference_seconds = time.perf_counter() - inference_started
        self._last_inference_seconds = float(inference_seconds)
        self._last_inference_frame = int(current_step) if current_step is not None else None
        print(f"[NovaCarterTIC-VLA] inference_complete latency={inference_seconds:.4f}s frame={current_step}")
        diagnostic_trace_path = os.getenv("PMG_ACTION_TRACE_PATH")
        if diagnostic_trace_path:
            diagnostic_inference = {
                "kind": "inference",
                "mode": os.getenv("STAGE2_DIAGNOSTIC_MODE", os.getenv("PMG_METHOD", "M0")),
                "simulation_frame": int(os.getenv("BENCHMARK_SIMULATION_FRAME_COUNT", "-1")),
                "inference_step": int(current_step) if current_step is not None else None,
                "monotonic_seconds": time.perf_counter(),
                "inference_latency_s": float(inference_seconds),
            }
            with open(diagnostic_trace_path, "a", encoding="utf-8") as diagnostic_trace:
                diagnostic_trace.write(__import__("json").dumps(diagnostic_inference, sort_keys=True) + "\\n")
        
        # Check AGAIN if backup started during model inference (model call takes time)'''
if text.count(inference_end) != 1:
    raise SystemExit("ERROR: expected TIC-VLA inference completion block was not found")
text = text.replace(inference_end, inference_end_with_latency)

# PMG context is updated by the benchmark frame loop.  The released behavior
# only reads TICVLA_INSTRUCTION during _configure_from_environment(), so make
# the refresh explicit immediately before each model query.  For M0 this is a
# no-op because the environment keeps the original instruction unchanged.
old_refresh = '''                    # Validate inputs before calling inference
                    if not sampled_paths:'''
new_refresh = '''                    # Refresh the instruction/goal before each model query.
                    # M1/M2/M3 update TICVLA_INSTRUCTION from the PMG frame loop;
                    # M0 retains its original instruction.
                    self._configure_from_environment()
                    # Validate inputs before calling inference
                    if not sampled_paths:'''
if text.count(old_refresh) != 1:
    raise SystemExit("ERROR: expected inference validation block was not found")
text = text.replace(old_refresh, new_refresh)

pmg_env_marker = '''            # Read instruction from environment variables
            instruction = os.getenv('TICVLA_INSTRUCTION')
            if instruction is not None:
                self._instruction = instruction
                carb.log_info(f"Instruction configured from environment: {instruction}")'''
pmg_env_block = pmg_env_marker + '''

            self._pmg_method = os.getenv("PMG_METHOD", "M0").upper()
            px, py = os.getenv("TICVLA_PMG_GOAL_X"), os.getenv("TICVLA_PMG_GOAL_Y")
            if px is not None and py is not None and self._pmg_method != "M0":
                self._pmg_goal_position = np.array([
                    float(px), float(py), float(os.getenv("TICVLA_PMG_GOAL_Z", "0.0"))
                ])
                self._pmg_score = float(os.getenv("TICVLA_PMG_SCORE", "0.0"))
                self._pmg_hard_valid = os.getenv("TICVLA_PMG_HARD_VALID", "0") == "1"
                self._pmg_stable = os.getenv("TICVLA_PMG_STABLE", "0") == "1"
                self._pmg_goal_distance = float(os.getenv("TICVLA_PMG_GOAL_DISTANCE", "inf"))
                self._pmg_scgp_enabled = os.getenv("TICVLA_PMG_SCGP_ENABLED", "0") == "1"
                self._pmg_scgp_allow = (
                    os.getenv("TICVLA_PMG_SCGP_ALLOW", "0") == "1"
                    if self._pmg_scgp_enabled else True
                )
                self._pmg_scgp_mode = os.getenv("TICVLA_PMG_SCGP_MODE", "fallback")
                self._pmg_scgp_reason = os.getenv("TICVLA_PMG_SCGP_REASON", "missing")
                try:
                    self._pmg_scgp_lease_until = int(os.getenv("TICVLA_PMG_SCGP_LEASE_UNTIL", "-1"))
                    self._pmg_scgp_stable_count = int(os.getenv("TICVLA_PMG_SCGP_STABLE_COUNT", "0"))
                    self._pmg_scgp_no_progress_ticks = int(os.getenv("TICVLA_PMG_SCGP_NO_PROGRESS_TICKS", "0"))
                except (TypeError, ValueError):
                    self._pmg_scgp_lease_until = -1
                    self._pmg_scgp_stable_count = 0
                    self._pmg_scgp_no_progress_ticks = 0
                wx, wy = os.getenv("TICVLA_PMG_WAYPOINT_X"), os.getenv("TICVLA_PMG_WAYPOINT_Y")
                self._pmg_route_enabled = os.getenv("TICVLA_PMG_ROUTE_ENABLED", "0") == "1"
                route_valid = os.getenv("TICVLA_PMG_ROUTE_VALID", "0") == "1"
                # A waypoint is actionable only when both flags agree.  The
                # publisher clears all route keys between frames, but this
                # second check makes the generated behavior fail closed if a
                # stale/inconsistent environment snapshot is ever observed.
                if self._pmg_route_enabled and route_valid and wx is not None and wy is not None:
                    try:
                        waypoint = np.array([
                            float(wx), float(wy), float(os.getenv("TICVLA_PMG_WAYPOINT_Z", "0.0"))
                        ], dtype=np.float64)
                        waypoint_index = int(os.getenv("TICVLA_PMG_WAYPOINT_INDEX", "-1"))
                        route_length = float(os.getenv("TICVLA_PMG_ROUTE_LENGTH", "inf"))
                        valid_waypoint = bool(np.all(np.isfinite(waypoint)) and waypoint_index >= 0 and np.isfinite(route_length) and route_length >= 0.0)
                    except (TypeError, ValueError):
                        valid_waypoint = False
                        waypoint = None
                        waypoint_index = -1
                        route_length = float("inf")
                    self._pmg_waypoint_position = waypoint if valid_waypoint else None
                    self._pmg_route_valid = valid_waypoint
                    self._pmg_route_reason = os.getenv("TICVLA_PMG_ROUTE_REASON", "astar") if valid_waypoint else "invalid_route_payload"
                    self._pmg_waypoint_index = waypoint_index if valid_waypoint else -1
                    self._pmg_route_length = route_length if valid_waypoint else float("inf")
                else:
                    self._pmg_waypoint_position = None
                    # Keep a distinct telemetry reason for the malformed
                    # combination ``ROUTE_VALID=1``/``ROUTE_ENABLED=0``;
                    # ordinary terminal deferrals retain their publisher
                    # supplied reason.
                    route_flag_mismatch = bool(route_valid and not self._pmg_route_enabled)
                    self._pmg_route_enabled = False
                    self._pmg_route_valid = False
                    self._pmg_route_reason = (
                        "route_enabled_flag_missing"
                        if route_flag_mismatch
                        else os.getenv("TICVLA_PMG_ROUTE_REASON", "route_unavailable")
                    )
                    self._pmg_waypoint_index = -1
                    self._pmg_route_length = float("inf")
            else:
                self._pmg_goal_position = None
                self._pmg_waypoint_position = None
                self._pmg_route_enabled = False
                self._pmg_route_valid = False
                self._pmg_route_reason = "no_pmg_payload"
                self._pmg_waypoint_index = -1
                self._pmg_route_length = float("inf")
                self._pmg_score = 0.0
                self._pmg_hard_valid = False
                self._pmg_stable = False
                self._pmg_goal_distance = float("inf")
                self._pmg_scgp_enabled = False
                self._pmg_scgp_allow = True
                self._pmg_scgp_mode = "disabled"
                self._pmg_scgp_reason = "no_pmg_payload"
                self._pmg_scgp_lease_until = -1
                self._pmg_scgp_stable_count = 0
                self._pmg_scgp_no_progress_ticks = 0
            # Resolve the fair global-policy/local-terminal hand-off after
            # every PMG payload refresh.  The gate uses candidate geometry
            # only; it never sees evaluator goal or success state.
            self._pmg_refresh_control_gate()'''
if text.count(pmg_env_marker) != 1:
    raise SystemExit("ERROR: expected behavior environment instruction block was not found")
text = text.replace(pmg_env_marker, pmg_env_block)

pmg_control_marker = '''            w_ff = 0.5 * v_cmd * kappa              # Reduced feedforward (was 1.0)
            w_fb = k_angular * self._yaw_err_filt   # Moderate feedback
            w_cmd = float(np.clip(w_ff + w_fb, -w_max, w_max))'''
pmg_control_block = pmg_control_marker + '''

        # Snapshot the released TIC-VLA command before any PMG intervention.
        # Every action tick contributes to the denominator of override_fraction;
        # this makes a global oracle hand-off auditable instead of looking like
        # an unqualified learned-policy improvement.
        policy_v_cmd = float(v_cmd)
        policy_w_cmd = float(w_cmd)
        pmg_control_source = "policy"
        pmg_override_applied = False
        pmg_gate_scope = "policy"
        pmg_gate_reason = str(getattr(self, "_pmg_gate_reason", "not_initialized"))
        pmg_control_policy = str(getattr(self, "_pmg_control_policy", "policy_only"))
        pmg_terminal_active = bool(getattr(self, "_pmg_terminal_active", False))
        pmg_terminal_eligible = bool(getattr(self, "_pmg_terminal_eligible", False))
        pmg_global_override = bool(getattr(self, "_pmg_global_override", False))
        pmg_candidate_valid = bool(getattr(self, "_pmg_candidate_valid", False))
        pmg_candidate_distance = getattr(self, "_pmg_candidate_distance_actual", None)
        simple_oracle_variant = (
            self._pmg_method == "M1"
            and os.getenv("STAGE2_M1_VARIANT", "").lower() == "simple_oracle"
        )
        # ``global`` is an explicit upper-bound profile.  The default
        # simple_oracle profile is ``terminal`` and therefore keeps TIC-VLA in
        # charge until the candidate enters the local hand-off radius.
        pmg_global_scope = bool(
            pmg_global_override
            and os.getenv("PMG_ACTION_ENABLED", "1") == "1"
        )
        pmg_terminal_scope = bool(
            pmg_terminal_eligible
            and pmg_control_policy in {"terminal", "recovery"}
            and os.getenv("PMG_ACTION_ENABLED", "1") == "1"
        )
        route_gate_block = bool(
            simple_oracle_variant
            and os.getenv("PMG_SIMPLE_ORACLE_REQUIRE_VALID_ROUTE", "1") == "1"
            and pmg_terminal_scope
            and not self._pmg_route_valid
        )
        # Causal paired evaluation: until PMG actually enters its terminal
        # scope with a valid route, use the exact paired M0 command. A route
        # rejection must not expose the independently sampled live policy.
        stage2_prefix_applied = False
        stage2_prefix_frame = int(os.getenv("BENCHMARK_SIMULATION_FRAME_COUNT", "-1"))
        stage2_prefix_command = None
        stage2_ticvla_v_cmd = float(v_cmd)
        stage2_ticvla_w_cmd = float(w_cmd)
        if (
            getattr(self, "_stage2_policy_prefix_enabled", False)
            and not pmg_global_scope
            and (not pmg_terminal_scope or route_gate_block)
            and not bool(getattr(self, "_pmg_checkpoint_released", False))
        ):
            if stage2_prefix_frame not in getattr(self, "_stage2_policy_prefix_by_frame", {}):
                raise RuntimeError(
                    "paired M1 action prefix has no M0 command for "
                    f"simulation frame {stage2_prefix_frame}"
                )
            stage2_prefix_command = self._stage2_policy_prefix_by_frame[stage2_prefix_frame]
            v_cmd, w_cmd = stage2_prefix_command
            policy_v_cmd = float(v_cmd)
            policy_w_cmd = float(w_cmd)
            stage2_prefix_applied = True
            pmg_control_source = "m0_policy_prefix"
            pmg_gate_reason = (
                "route_infeasible_policy_fallback"
                if route_gate_block else "paired_prefix_before_terminal"
            )
        # A route planner supplies an immediate collision-free waypoint while
        # the semantic PMG candidate remains the terminal goal.  Older
        # behavior scripts ignore these fields and continue to use the goal.
        pmg_control_position = (
            self._pmg_waypoint_position
            if bool(getattr(self, "_pmg_route_valid", False))
            and getattr(self, "_pmg_waypoint_position", None) is not None
            else self._pmg_goal_position
        )
        # A one-point ``direct_visible`` plan is a direct-goal controller, not
        # an A* waypoint route.  Treating it as a waypoint route used to send
        # local stalls into ``route_stall_policy_fallback`` indefinitely,
        # masking the bounded direct-goal detour branch below.  The e050
        # trace shows exactly this failure: route_reason=direct_visible,
        # waypoint_index=0, then 999 policy fallbacks after 42 terminal
        # actions.  Only a non-direct plan with an actual intermediate
        # waypoint should use the route-specific fallback semantics.
        route_reason = str(getattr(self, "_pmg_route_reason", ""))
        route_waypoint_active = bool(
            getattr(self, "_pmg_route_valid", False)
            and getattr(self, "_pmg_waypoint_position", None) is not None
            and route_reason not in {"direct_visible", "no_static_obstacles", "same_free_cell"}
            and not (
                bool(getattr(self, "_pmg_scgp_enabled", False))
                and not bool(getattr(self, "_pmg_scgp_allow", False))
            )
        )
        pmg_route_alignment_hold = False
        pmg_route_heading_error_rad = None
        route_waypoint_key = None
        route_alignment_hold = False
        if route_waypoint_active and self._robot_position is not None:
            route_goal_key = (
                tuple(round(float(value), 3) for value in self._pmg_goal_position[:2])
                if self._pmg_goal_position is not None
                else (None, None)
            )
            route_waypoint_key = (
                int(getattr(self, "_pmg_waypoint_index", -1)),
                round(float(self._pmg_waypoint_position[0]), 3),
                round(float(self._pmg_waypoint_position[1]), 3),
                *route_goal_key,
            )
            route_distance = float(np.linalg.norm(
                np.asarray(self._pmg_waypoint_position[:2], dtype=np.float64)
                - np.asarray(self._robot_position[:2], dtype=np.float64)
            ))
            try:
                progress_epsilon = max(0.05, float(os.getenv("PMG_ROUTE_PROGRESS_EPS_M", "0.05")))
            except (TypeError, ValueError):
                progress_epsilon = 0.05
            if route_waypoint_key != getattr(self, "_pmg_progress_release_key", None):
                self._pmg_progress_release_key = route_waypoint_key
                self._pmg_progress_release_best_distance = route_distance
                self._pmg_progress_release_attempted = False
                self._pmg_route_stall_latched = False
                self._pmg_progress_stall_inferences = 0
                self._pmg_stall_frames = 0
                self._pmg_detour_frames = 0
            elif route_distance < float(getattr(self, "_pmg_progress_release_best_distance", float("inf"))) - progress_epsilon:
                # Only a new measured best distance, not elapsed time or a
                # small oscillation, earns another bounded release attempt.
                self._pmg_progress_release_best_distance = route_distance
                self._pmg_progress_release_attempted = False
                self._pmg_route_stall_latched = False
                self._pmg_progress_stall_inferences = 0
            route_delta = np.asarray(self._pmg_waypoint_position[:2], dtype=np.float64) - np.asarray(self._robot_position[:2], dtype=np.float64)
            quaternion = getattr(self, "_robot_quaternion", None)
            if quaternion is not None and len(quaternion) >= 4:
                qw, qx, qy, qz = (float(quaternion[index]) for index in range(4))
                route_heading = math.atan2(
                    2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz),
                )
            else:
                route_heading = float(getattr(self, "_robot_heading", getattr(self, "_current_heading", 0.0)))
            route_x_body = math.cos(route_heading) * float(route_delta[0]) + math.sin(route_heading) * float(route_delta[1])
            route_y_body = -math.sin(route_heading) * float(route_delta[0]) + math.cos(route_heading) * float(route_delta[1])
            route_heading_error = math.atan2(route_y_body, route_x_body)
            if abs(route_heading_error) > 2.6:
                route_heading_error = math.copysign(2.6, route_heading_error)
            try:
                align_release = max(0.0, float(os.getenv("PMG_ROUTE_ALIGN_RELEASE_RAD", "0.35")))
            except (TypeError, ValueError):
                align_release = 0.35
            try:
                align_hold = max(align_release, float(os.getenv("PMG_ROUTE_ALIGN_HOLD_RAD", "0.55")))
            except (TypeError, ValueError):
                align_hold = max(align_release, 0.55)
            route_alignment_hold = bool(
                abs(route_heading_error) > (
                    align_release if bool(getattr(self, "_pmg_route_aligning", False)) else align_hold
                )
            )
        elif not route_waypoint_active:
            # A failed/deferred route invalidates any release budget tied to
            # the previous waypoint.  Without this reset, a rebuilt route
            # with the same key could inherit a stale policy-fallback latch.
            self._pmg_progress_waypoint_key = None
            self._pmg_progress_best_distance = float("inf")
            self._pmg_progress_stall_inferences = 0
            self._pmg_progress_release_key = None
            self._pmg_progress_release_best_distance = float("inf")
            self._pmg_progress_release_attempted = False
            self._pmg_route_stall_latched = False
        # A direct goal vector cannot reason about an obstacle absent from the
        # compact PMG. Allow one bounded detour per direct candidate, then
        # yield to policy if PMG again commands motion without moving. The
        # key and watchdog use only PMG geometry and measured body motion.
        pmg_recovery_suppressed = False
        if (
            simple_oracle_variant
            and (pmg_global_scope or pmg_terminal_scope)
            and pmg_control_position is not None
            and self._robot_position is not None
            and os.getenv("PMG_ACTION_ENABLED", "1") == "1"
        ):
            direct_route_active = bool(
                getattr(self, "_pmg_route_valid", False)
                and route_reason in {"direct_visible", "no_static_obstacles", "same_free_cell"}
            )
            direct_stall_key = (
                tuple(round(float(value), 3) for value in self._pmg_goal_position[:2])
                if self._pmg_goal_position is not None else None
            )
            if direct_stall_key != getattr(self, "_pmg_direct_stall_key", None):
                self._pmg_direct_stall_key = direct_stall_key
                self._pmg_direct_detour_attempted = False
                self._pmg_direct_stall_latched = False
                self._pmg_stall_frames = 0
                self._pmg_detour_frames = 0
            elif route_waypoint_active:
                # The direct-goal latch is irrelevant on A*, but its reset
                # must not erase a live route detour countdown every tick.
                self._pmg_direct_detour_attempted = False
                self._pmg_direct_stall_latched = False
            measured_speed = float(np.linalg.norm(np.asarray(self._robot_velocity[:2], dtype=np.float64)))
            applied_speed = abs(float(getattr(self, "_cmd_v", 0.0)))
            stall_speed = float(os.getenv("PMG_STALL_SPEED", "0.05"))
            commanded_speed = float(os.getenv("PMG_STALL_COMMAND_SPEED", "0.16"))
            stall_window = max(1, int(os.getenv("PMG_STALL_WINDOW_INFERENCES", "8")))
            detour_cycles = max(0, int(getattr(self, "_pmg_detour_frames", 0)))
            if route_waypoint_active and self._pmg_route_stall_latched:
                self._pmg_stall_frames = 0
                self._pmg_detour_frames = 0
                pmg_recovery_suppressed = True
                pmg_gate_reason = "route_progress_stalled_policy_fallback"
            elif route_waypoint_active and route_alignment_hold:
                # Rotate-first is intentional zero translation, and a route
                # detour must not interrupt the current heading correction.
                self._pmg_stall_frames = 0
                self._pmg_detour_frames = 0
            elif self._pmg_direct_stall_latched and not route_waypoint_active:
                self._pmg_stall_frames = 0
                self._pmg_detour_frames = 0
                pmg_recovery_suppressed = True
                pmg_gate_reason = "direct_route_stalled_policy_fallback"
            elif not getattr(self, "_pmg_route_valid", False) and not pmg_global_scope:
                # A stale recovery window cannot override a route gate that
                # has since failed closed.
                self._pmg_stall_frames = 0
                self._pmg_detour_frames = 0
                pmg_recovery_suppressed = True
                pmg_gate_reason = "route_infeasible_policy_fallback"
            elif detour_cycles > 0:
                self._pmg_detour_frames = detour_cycles - 1
                self._pmg_stall_frames = 0
                pmg_recovery_suppressed = True
            else:
                # Policy and detour ticks cannot count toward a fresh PMG
                # stall. Require consecutive PMG translation commands after
                # the recovery window before deciding that it failed.
                last_source = str(getattr(self, "_pmg_last_control_source", "policy"))
                if (
                    last_source in {"pmg_terminal", "pmg_global"}
                    and measured_speed <= stall_speed
                    and applied_speed >= commanded_speed
                ):
                    self._pmg_stall_frames += 1
                else:
                    self._pmg_stall_frames = 0
                if self._pmg_stall_frames >= stall_window:
                    self._pmg_stall_frames = 0
                    if direct_route_active and self._pmg_direct_detour_attempted:
                        self._pmg_direct_stall_latched = True
                        pmg_recovery_suppressed = True
                        pmg_gate_reason = "direct_route_stalled_policy_fallback"
                    elif route_waypoint_active and self._pmg_progress_release_attempted:
                        self._pmg_route_stall_latched = True
                        self._pmg_detour_frames = 0
                        pmg_recovery_suppressed = True
                        pmg_gate_reason = "route_progress_stalled_policy_fallback"
                    else:
                        if direct_route_active:
                            self._pmg_direct_detour_attempted = True
                        if route_waypoint_active:
                            self._pmg_progress_release_attempted = True
                        self._pmg_detour_frames = max(
                            1, int(os.getenv("PMG_DETOUR_INFERENCES", "24"))
                        )
                        # Pick the detour side from the PMG candidate's
                        # body-frame bearing. If it is straight ahead,
                        # alternate sides between independent stalls.
                        delta = pmg_control_position[:2] - self._robot_position[:2]
                        quaternion = getattr(self, "_robot_quaternion", None)
                        if quaternion is not None and len(quaternion) >= 4:
                            qw, qx, qy, qz = (float(quaternion[index]) for index in range(4))
                            heading = math.atan2(
                                2.0 * (qw * qz + qx * qy),
                                1.0 - 2.0 * (qy * qy + qz * qz),
                            )
                        else:
                            heading = float(getattr(self, "_current_heading", 0.0))
                        y_body = -math.sin(heading) * float(delta[0]) + math.cos(heading) * float(delta[1])
                        if abs(y_body) > 0.20:
                            self._pmg_detour_side = 1.0 if y_body > 0.0 else -1.0
                        else:
                            self._pmg_detour_side = -float(getattr(self, "_pmg_detour_side", 1.0))
                        pmg_recovery_suppressed = True
                        print(
                            f"[PMG] local_stall detected speed={measured_speed:.3f} "
                            f"cmd={applied_speed:.3f}; recovery for "
                            f"{self._pmg_detour_frames} inference cycles "
                            f"side={self._pmg_detour_side:+.0f}"
                        )
        elif simple_oracle_variant:
            # Do not accumulate a pre-terminal stall window.  Otherwise a
            # candidate entering the local radius later could inherit stale
            # detour state from the global TIC-VLA route.
            self._pmg_stall_frames = 0
            self._pmg_detour_frames = 0
            self._pmg_direct_stall_key = None
            self._pmg_direct_detour_attempted = False
            self._pmg_direct_stall_latched = False
        if (
            self._pmg_method in {"M1", "M2", "M3"}
            and os.getenv("PMG_ACTION_ENABLED", "1") == "1"
            and pmg_control_position is not None
            and self._pmg_hard_valid
            and (simple_oracle_variant or self._pmg_stable)
            and (pmg_global_scope or pmg_terminal_scope)
            and not pmg_recovery_suppressed
            and self._pmg_score >= float(
                os.getenv("PMG_SIMPLE_ORACLE_MIN_SCORE", "0.0")
                if simple_oracle_variant
                else os.getenv("PMG_RECOVERY_MIN_SCORE", "0.92")
            )
            and self._robot_position is not None
            and (
                # A simple-oracle run is globally gated only when explicitly
                # requested.  Its default terminal profile reaches this block
                # solely after the semantic candidate enters the local scope.
                (pmg_global_scope or pmg_terminal_scope)
                if simple_oracle_variant
                else self._pmg_goal_distance <= float(os.getenv("PMG_RECOVERY_MAX_DISTANCE", "4.0"))
            )
        ):
            delta = pmg_control_position[:2] - self._robot_position[:2]
            distance = float(np.linalg.norm(delta))
            if route_waypoint_active:
                # During rotate-first alignment the controller is expected to
                # make little or no translational progress.  Counting those
                # frames as a route stall triggers the detour watchdog and
                # repeatedly interrupts heading convergence.  The alignment
                # current heading is evaluated before this watchdog, so even
                # the first rotate-only tick cannot start a progress release.
                # Resume detection after alignment permits translation.
                if route_alignment_hold:
                    self._pmg_progress_stall_inferences = 0
                    self._pmg_progress_best_distance = distance
                else:
                    waypoint_key = route_waypoint_key
                    if waypoint_key != getattr(self, "_pmg_progress_waypoint_key", None):
                        self._pmg_progress_waypoint_key = waypoint_key
                        self._pmg_progress_best_distance = distance
                        self._pmg_progress_stall_inferences = 0
                    else:
                        if distance < float(getattr(self, "_pmg_progress_best_distance", float("inf"))) - progress_epsilon:
                            self._pmg_progress_best_distance = distance
                            self._pmg_progress_stall_inferences = 0
                        else:
                            self._pmg_progress_stall_inferences = int(
                                getattr(self, "_pmg_progress_stall_inferences", 0)
                            ) + 1
                    progress_window = max(
                        1,
                        int(os.getenv("PMG_ROUTE_PROGRESS_STALL_INFERENCES", "12")),
                    )
                    if self._pmg_progress_stall_inferences >= progress_window:
                        self._pmg_progress_stall_inferences = 0
                        self._pmg_stall_frames = 0
                        if self._pmg_progress_release_attempted:
                            # Same waypoint and no new measured best after
                            # one release: stop repeating the unverified arc.
                            self._pmg_route_stall_latched = True
                            self._pmg_detour_frames = 0
                            pmg_recovery_suppressed = True
                            pmg_gate_reason = "route_progress_stalled_policy_fallback"
                        else:
                            self._pmg_progress_release_attempted = True
                            self._pmg_detour_frames = max(
                                1,
                                int(os.getenv("PMG_ROUTE_PROGRESS_RELEASE_INFERENCES", "12")),
                            )
                            self._pmg_detour_side = -float(
                                getattr(self, "_pmg_detour_side", 1.0)
                            )
                            self._pmg_progress_release_count = int(
                                getattr(self, "_pmg_progress_release_count", 0)
                            ) + 1
                            pmg_recovery_suppressed = True
                            print(
                                f"[PMG] route_progress_stall distance={distance:.3f} "
                                f"waypoint={waypoint_key[0]} release={self._pmg_detour_frames}"
                            )
            else:
                self._pmg_progress_waypoint_key = None
                self._pmg_progress_best_distance = float("inf")
                self._pmg_progress_stall_inferences = 0
            simple_stop_radius = float(os.getenv("PMG_SIMPLE_ORACLE_STOP_RADIUS", "0.60"))
            # Intermediate A* points are transit states, not terminal goals.
            # Reusing the semantic 0.60 m stop radius can halt Carter before
            # the runtime's conservative waypoint-progress guard observes a
            # genuinely free pose (the e041 escape point stopped at ~0.57 m).
            # Keep the final semantic candidate unchanged, but let an
            # intermediate route point be approached more closely.  The
            # radius is bounded and only affects the explicit route-control
            # path; it does not relax planner clearance or segment checks.
            route_stop_radius = simple_stop_radius
            if route_waypoint_active:
                try:
                    route_stop_radius = max(
                        0.10,
                        min(0.45, float(os.getenv("PMG_ROUTE_WAYPOINT_STOP_RADIUS", "0.20"))),
                    )
                except (TypeError, ValueError):
                    route_stop_radius = 0.20
            is_intermediate_route_point = bool(
                route_waypoint_active
                and getattr(self, "_pmg_goal_position", None) is not None
                and float(
                    np.linalg.norm(
                        np.asarray(self._pmg_waypoint_position[:2], dtype=np.float64)
                        - np.asarray(self._pmg_goal_position[:2], dtype=np.float64)
                    )
                ) > 0.05
            )
            if is_intermediate_route_point:
                simple_stop_radius = route_stop_radius
            # Keep this default defined even after the controller enters the
            # stop-radius branch; the terminal fallback below also consults
            # it when no translational command is needed.
            if distance >= simple_stop_radius:
                # Read the robot's world yaw from the USD quaternion.  The
                # legacy ``_current_heading`` is derived from a transposed USD
                # matrix in the Isaac-6 compatibility path (its sign is
                # reversed for yaw), which sends a rear target into an endless
                # turn/stop loop.  Quaternions are stored as [w, x, y, z].
                quaternion = getattr(self, "_robot_quaternion", None)
                if quaternion is not None and len(quaternion) >= 4:
                    qw, qx, qy, qz = (float(quaternion[index]) for index in range(4))
                    heading = math.atan2(
                        2.0 * (qw * qz + qx * qy),
                        1.0 - 2.0 * (qy * qy + qz * qz),
                    )
                else:
                    heading = float(getattr(self, "_robot_heading", getattr(self, "_current_heading", 0.0)))
                x_body = math.cos(heading) * float(delta[0]) + math.sin(heading) * float(delta[1])
                y_body = -math.sin(heading) * float(delta[0]) + math.cos(heading) * float(delta[1])
                # Preserve the full +/-pi bearing when the candidate is behind
                # the robot. Clipping x_body to a positive epsilon collapses a
                # rear target to +/-pi/2 and can make recovery spin forever.
                goal_yaw = math.atan2(y_body, x_body)
                # goal_yaw is already the candidate bearing in the robot body
                # frame. _yaw_err_filt belongs to the learned waypoint
                # controller and is not the robot heading, so mixing it here
                # can suppress recovery for side/rear candidates.
                heading_error = goal_yaw
                # atan2 can alternate between +pi and -pi when a target is
                # directly behind the robot. Preserve the shortest-turn sign
                # when clamping the narrow rear-target band; mapping both
                # signs to +2.6 makes the controller cross +/-pi and
                # oscillate in place indefinitely.
                if abs(heading_error) > 2.6:
                    heading_error = math.copysign(2.6, heading_error)
                pmg_route_heading_error_rad = float(heading_error)
                # A point-goal controller that starts translating through a
                # large heading change cuts inside A* corners.  On Carter this
                # can wedge the rectangular footprint against a wall even
                # though both occupancy-grid endpoints are free.  Use a small
                # hysteretic rotate-first gate for route waypoints; it depends
                # only on robot pose and the PMG waypoint, never evaluator
                # state.  Direct/terminal goal behavior remains unchanged.
                if route_waypoint_active:
                    pmg_route_alignment_hold = bool(route_alignment_hold)
                    self._pmg_route_aligning = pmg_route_alignment_hold
                else:
                    self._pmg_route_aligning = False
                recovery_stall_speed = float(os.getenv("PMG_RECOVERY_STALL_SPEED", "0.18"))
                recovery_min_heading = float(os.getenv("PMG_RECOVERY_MIN_HEADING", "1.05"))
                recovery_needed = abs(policy_v_cmd) <= recovery_stall_speed or abs(heading_error) >= recovery_min_heading
                terminal_trigger = bool(
                    pmg_global_scope
                    or (
                        pmg_terminal_scope
                        and pmg_control_policy == "terminal"
                        and os.getenv("PMG_TERMINAL_TRIGGER", "handoff").lower() not in {"recovery", "stall"}
                    )
                    or (
                        pmg_terminal_scope
                        and pmg_control_policy == "recovery"
                        and recovery_needed
                    )
                )
                # A semantic candidate is not enough to justify taking over
                # the robot. In the fair simple-oracle M1 profile, require a
                # currently validated route; otherwise yield to TIC-VLA.
                # Global oracle runs remain an explicitly separate ablation.
                if route_gate_block:
                    terminal_trigger = False
                    pmg_gate_reason = "route_infeasible_policy_fallback"
                if terminal_trigger:
                    recovery_v = (
                        0.0
                        if pmg_route_alignment_hold
                        else float(np.clip(0.50 * max(x_body, 0.0), 0.0, v_max * 0.50))
                    )
                    # The Isaac-6 compatibility runner applies the angular
                    # command with the same sign as the quaternion-derived
                    # body-frame bearing.  The paired e041 trace is the
                    # contract here: positive bearing increased yaw and
                    # reached the first route waypoint, while negating it
                    # kept the robot rotating at the start cell.  Keep the
                    # signed error in telemetry so a future simulator change
                    # cannot silently invert this control law.
                    recovery_w = float(np.clip(heading_error, -w_max * 0.65, w_max * 0.65))
                    if pmg_global_scope and simple_oracle_variant:
                        blend_value = os.getenv("PMG_SIMPLE_ORACLE_BLEND", "1.0")
                    elif pmg_terminal_scope and simple_oracle_variant:
                        blend_value = os.getenv("PMG_TERMINAL_BLEND", "1.0")
                    else:
                        blend_value = os.getenv("PMG_RECOVERY_BLEND", "0.35")
                    try:
                        recovery_blend = float(np.clip(float(blend_value), 0.0, 1.0))
                    except (TypeError, ValueError):
                        recovery_blend = 1.0 if simple_oracle_variant else 0.35
                    v_cmd = (1.0 - recovery_blend) * v_cmd + recovery_blend * recovery_v
                    w_cmd = (1.0 - recovery_blend) * w_cmd + recovery_blend * recovery_w
                    pmg_override_applied = True
                    pmg_gate_scope = "global" if pmg_global_scope else "terminal"
                    pmg_control_source = "pmg_global" if pmg_global_scope else "pmg_terminal"
                    pmg_gate_reason = (
                        "route_heading_alignment"
                        if pmg_route_alignment_hold
                        else ("explicit_global_override" if pmg_global_scope else "terminal_handoff")
                    )
                    print(f"[PMG] bounded_recovery scope={pmg_gate_scope} distance={distance:.2f} heading_error={heading_error:.2f} v={v_cmd:.3f} w={w_cmd:.3f}")
            elif (pmg_global_scope or (pmg_terminal_scope and pmg_control_policy == "terminal")) and not route_gate_block:
                # Arrival/release belongs to the semantic PMG candidate, not
                # an intermediate route waypoint.  Using ``distance`` here
                # released authority as soon as Carter reached a coarse A*
                # cell (e053: semantic distance was still about 1.35 m),
                # turning a transit point into a false checkpoint arrival.
                semantic_release_distance = float("inf")
                if self._robot_position is not None and self._pmg_goal_position is not None:
                    semantic_release_distance = float(
                        np.linalg.norm(
                            np.asarray(self._pmg_goal_position[:2], dtype=np.float64)
                            - np.asarray(self._robot_position[:2], dtype=np.float64)
                        )
                    )
                release_on_arrival = bool(
                    simple_oracle_variant
                    and os.getenv("PMG_SIMPLE_ORACLE_RELEASE_ON_ARRIVAL", "0").lower()
                    in {"1", "true", "yes", "on"}
                    and semantic_release_distance <= simple_stop_radius
                )
                if release_on_arrival:
                    self._pmg_checkpoint_released = True
                    self._pmg_terminal_active = False
                    self._pmg_terminal_eligible = False
                    pmg_terminal_scope = False
                    pmg_global_scope = False
                    pmg_gate_scope = "policy"
                    pmg_control_source = "policy"
                    pmg_gate_reason = "checkpoint_arrival_release"
                else:
                    v_cmd = 0.0
                    w_cmd = 0.0
                    pmg_override_applied = True
                    pmg_gate_scope = "global" if pmg_global_scope else "terminal"
                    pmg_control_source = "pmg_global" if pmg_global_scope else "pmg_terminal"
                    pmg_gate_reason = "goal_stop"

        # A route-progress latch may be raised inside the controller above,
        # after the ordinary terminal action was computed. Restore the policy
        # command before considering any bounded route detour.
        if route_waypoint_active and self._pmg_route_stall_latched:
            v_cmd = policy_v_cmd
            w_cmd = policy_w_cmd
            pmg_override_applied = False
            pmg_gate_scope = "policy"
            pmg_control_source = "policy"
            pmg_gate_reason = "route_progress_stalled_policy_fallback"
        route_stall_policy_fallback = bool(
            pmg_recovery_suppressed
            and route_waypoint_active
            and not self._pmg_route_stall_latched
        )
        if route_stall_policy_fallback:
            # A route stall is a PMG-controller event, not permission to hand
            # the robot back to the unconstrained TIC-VLA policy indefinitely.
            # In e081 the old fallback began at frame 1729 while waypoint 0
            # was still 1.21 m away; the policy then kept the route index at 0
            # for the rest of the horizon.  Keep the bounded detour local to
            # the geometry-checked route so the robot can escape the coarse
            # cell and re-enter the normal waypoint controller.
            try:
                route_detour_v = float(
                    np.clip(
                        float(os.getenv("PMG_ROUTE_DETOUR_SPEED", "0.12")),
                        0.0,
                        v_max * 0.35,
                    )
                )
            except (TypeError, ValueError):
                route_detour_v = min(0.12, v_max * 0.35)
            try:
                route_detour_w = float(
                    np.clip(
                        float(os.getenv("PMG_ROUTE_DETOUR_TURN_RATE", "0.75")),
                        0.0,
                        w_max * 0.75,
                    )
                )
            except (TypeError, ValueError):
                route_detour_w = min(0.75, w_max * 0.75)
            v_cmd = route_detour_v
            w_cmd = float(
                np.clip(
                    float(getattr(self, "_pmg_detour_side", 1.0)) * route_detour_w,
                    -w_max * 0.75,
                    w_max * 0.75,
                )
            )
            pmg_override_applied = True
            pmg_gate_scope = "global" if pmg_global_scope else "terminal"
            pmg_control_source = "pmg_detour"
            pmg_gate_reason = "route_stall_detour"
        if (
            pmg_recovery_suppressed
            and not route_stall_policy_fallback
            and simple_oracle_variant
            and (pmg_global_scope or pmg_terminal_scope)
            and (pmg_global_scope or bool(getattr(self, "_pmg_route_valid", False)))
            and not bool(getattr(self, "_pmg_direct_stall_latched", False))
            and not bool(getattr(self, "_pmg_route_stall_latched", False))
        ):
            policy_cycles = max(1, int(os.getenv("PMG_POLICY_RECOVERY_INFERENCES", "12")))
            if int(getattr(self, "_pmg_detour_frames", 0)) <= policy_cycles:
                detour_v = float(os.getenv("PMG_DETOUR_SPEED", "0.12"))
                detour_w = float(os.getenv("PMG_DETOUR_TURN_RATE", "0.75"))
                v_cmd = float(np.clip(detour_v, 0.0, v_max * 0.35))
                w_cmd = float(np.clip(float(getattr(self, "_pmg_detour_side", 1.0)) * detour_w, -w_max * 0.75, w_max * 0.75))
                pmg_override_applied = True
                pmg_gate_scope = "global" if pmg_global_scope else "terminal"
                pmg_control_source = "pmg_detour"
                pmg_gate_reason = "local_stall_detour"
                print(
                    f"[PMG] local_detour remaining={int(getattr(self, '_pmg_detour_frames', 0))} "
                    f"v={v_cmd:.3f} w={w_cmd:.3f}"
                )

        # Persist one auditable control-source decision per action tick.  The
        # denominator includes M0/TIC-VLA policy ticks as well as PMG ticks;
        # ``pmg_override_fraction`` therefore cannot hide a start-to-goal
        # oracle takeover behind a sparse PMG trace.
        self._pmg_action_ticks = int(getattr(self, "_pmg_action_ticks", 0)) + 1
        if pmg_override_applied:
            self._pmg_override_ticks = int(getattr(self, "_pmg_override_ticks", 0)) + 1
            if pmg_control_source == "pmg_detour":
                self._pmg_detour_override_ticks = int(getattr(self, "_pmg_detour_override_ticks", 0)) + 1
            elif pmg_gate_scope == "global":
                self._pmg_global_override_ticks = int(getattr(self, "_pmg_global_override_ticks", 0)) + 1
            elif pmg_gate_scope == "terminal":
                self._pmg_terminal_override_ticks = int(getattr(self, "_pmg_terminal_override_ticks", 0)) + 1
        else:
            self._pmg_policy_ticks = int(getattr(self, "_pmg_policy_ticks", 0)) + 1
        self._pmg_last_control_source = str(pmg_control_source)
        self._pmg_last_override = bool(pmg_override_applied)
        pmg_override_fraction = float(
            self._pmg_override_ticks / max(1, self._pmg_action_ticks)
        )

        # Action output trace is intentionally placed after the PMG gate.  In
        # the M3 ablation, PMG may update its graph but this checksum sequence
        # must follow the same command path as M0.
        diagnostic_trace_path = os.getenv("PMG_ACTION_TRACE_PATH")
        if diagnostic_trace_path:
            diagnostic_output = {
                "kind": "output",
                "mode": os.getenv("STAGE2_DIAGNOSTIC_MODE", os.getenv("PMG_METHOD", "M0")),
                "simulation_frame": int(os.getenv("BENCHMARK_SIMULATION_FRAME_COUNT", "-1")),
                "monotonic_seconds": time.perf_counter(),
                "control_policy": str(pmg_control_policy),
                "policy_v_cmd": float(policy_v_cmd),
                "policy_w_cmd": float(policy_w_cmd),
                "stage2_paired_action_prefix_enabled": bool(
                    getattr(self, "_stage2_policy_prefix_enabled", False)
                ),
                "stage2_prefix_applied": bool(stage2_prefix_applied),
                "stage2_prefix_frame": int(stage2_prefix_frame),
                "stage2_prefix_command": (
                    [float(stage2_prefix_command[0]), float(stage2_prefix_command[1])]
                    if stage2_prefix_command is not None else None
                ),
                "stage2_ticvla_command": [
                    float(stage2_ticvla_v_cmd), float(stage2_ticvla_w_cmd)
                ],
                "v_cmd": float(v_cmd),
                "w_cmd": float(w_cmd),
                "pmg_override": bool(pmg_override_applied),
                "pmg_override_fraction": pmg_override_fraction,
                "pmg_terminal_active": bool(pmg_terminal_active),
                "pmg_terminal_eligible": bool(pmg_terminal_eligible),
                "pmg_global_scope": bool(pmg_global_scope),
                "pmg_gate_reason": str(pmg_gate_reason),
                "pmg_candidate_valid": bool(pmg_candidate_valid),
                "pmg_candidate_distance_m": (
                    float(pmg_candidate_distance) if pmg_candidate_distance is not None else None
                ),
                "pmg_semantic_candidate_distance_m": (
                    float(getattr(self, "_pmg_semantic_candidate_distance_actual", pmg_candidate_distance))
                    if getattr(self, "_pmg_semantic_candidate_distance_actual", pmg_candidate_distance) is not None
                    else None
                ),
                "pmg_control_candidate_distance_m": (
                    float(np.linalg.norm(
                        np.asarray(pmg_control_position[:2], dtype=np.float64)
                        - np.asarray(self._robot_position[:2], dtype=np.float64)
                    ))
                    if pmg_control_position is not None and self._robot_position is not None else None
                ),
                "pmg_candidate": (
                    [float(value) for value in pmg_control_position[:3]]
                    if pmg_control_position is not None else None
                ),
                "pmg_control_candidate": (
                    [float(value) for value in pmg_control_position[:3]]
                    if pmg_control_position is not None else None
                ),
                "pmg_semantic_candidate": (
                    [float(value) for value in self._pmg_goal_position[:3]]
                    if self._pmg_goal_position is not None else None
                ),
                "pmg_route_valid": bool(getattr(self, "_pmg_route_valid", False)),
                "pmg_route_enabled": os.getenv("TICVLA_PMG_ROUTE_ENABLED", "0") == "1",
                "pmg_route_reason": str(getattr(self, "_pmg_route_reason", "not_provided")),
                "pmg_scgp_enabled": bool(getattr(self, "_pmg_scgp_enabled", False)),
                "pmg_scgp_allow": bool(getattr(self, "_pmg_scgp_allow", True)),
                "pmg_scgp_mode": str(getattr(self, "_pmg_scgp_mode", "disabled")),
                "pmg_scgp_reason": str(getattr(self, "_pmg_scgp_reason", "disabled")),
                "pmg_scgp_lease_until": int(getattr(self, "_pmg_scgp_lease_until", -1)),
                "pmg_scgp_stable_count": int(getattr(self, "_pmg_scgp_stable_count", 0)),
                "pmg_scgp_no_progress_ticks": int(getattr(self, "_pmg_scgp_no_progress_ticks", 0)),
                "pmg_waypoint_index": int(getattr(self, "_pmg_waypoint_index", -1)),
                "pmg_route_alignment_hold": bool(pmg_route_alignment_hold),
                "pmg_route_heading_error_rad": pmg_route_heading_error_rad,
                "pmg_route_stall_policy_fallback": bool(route_stall_policy_fallback),
                "pmg_direct_stall_latched": bool(getattr(self, "_pmg_direct_stall_latched", False)),
                "pmg_route_stall_latched": bool(getattr(self, "_pmg_route_stall_latched", False)),
                "pmg_route_progress_stall_inferences": int(
                    getattr(self, "_pmg_progress_stall_inferences", 0)
                ),
                "pmg_route_progress_release_count": int(
                    getattr(self, "_pmg_progress_release_count", 0)
                ),
                "pmg_control_source": str(pmg_control_source),
                "pmg_gate_scope": str(pmg_gate_scope),
                "recovery": bool(pmg_override_applied),
            }
            diagnostic_output["last_inference_latency_s"] = (
                float(getattr(self, "_last_inference_seconds", float("nan")))
                if np.isfinite(float(getattr(self, "_last_inference_seconds", float("nan"))))
                else None
            )
            diagnostic_output["checksum"] = __import__("hashlib").sha256(
                __import__("json").dumps(
                    [diagnostic_output["v_cmd"], diagnostic_output["w_cmd"]],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            with open(diagnostic_trace_path, "a", encoding="utf-8") as diagnostic_trace:
                diagnostic_trace.write(__import__("json").dumps(diagnostic_output, sort_keys=True) + "\\n")'''
if text.count(pmg_control_marker) != 1:
    raise SystemExit("ERROR: expected pure-pursuit command block was not found")
text = text.replace(pmg_control_marker, pmg_control_block)
path.write_text(text, encoding="utf-8")
PY

  shim_source="${PROJECT_MAIN:-${ticvla_work_root}/project/main}/autodl/scripts/21_robot_navigation_manager_isaac6.py"
  if [[ ! -f "${shim_source}" ]]; then
    echo "ERROR: Isaac Sim 6 navigation shim is missing: ${shim_source}" >&2
    exit 6
  fi
  cp "${shim_source}" "${runtime_dir}/robot_navigation_manager.py"

  diff -u "${ticvla_repo_dir}/DynaNav/behavior/nova_carter_test_ticvla.py" \
    "${behavior_script}" > "${manifest_dir}/${run_id}_isaac6_behavior.patch" || true
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ISAAC_SIM_ROOT="${isaac_sim_root}"
export ISAAC_SIM_PYTHON="${isaac_sim_python}"
export TICVLA_DYNANAV_ROOT="${ticvla_repo_dir}/DynaNav"
export PMG_METRICS_ROOT="${pmg_metrics_root}"
export PYTHONPATH="${ticvla_repo_dir}/DynaNav:${ticvla_repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
# Isaac's tokenizer stack uses rayon internally.  Keeping all tokenizer and
# BLAS thread pools bounded makes paired runs reproducible and avoids the
# long-run tokenizers/ABI crash observed in the previous e041 attempt.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
if [[ -f /etc/vulkan/icd.d/my_nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "ISAAC_SIM_PYTHON=${ISAAC_SIM_PYTHON}"
echo "TICVLA_DYNANAV_ROOT=${TICVLA_DYNANAV_ROOT}"
echo "HEADLESS_BENCHMARK=${headless_script}"
echo "BENCHMARK_RUN_ID=${run_id}"
echo "ISAAC6_COMPAT=${isaac6_compat}"
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-<loader-default>}"

# Cheap code-generation preflight for Stage 2.  This stops after all locked
# benchmark/behavior patches and the Isaac-6 navigation shim have been
# materialized, but before starting the GPU monitor or Isaac Sim.  It lets a
# run gate prove that the exact generated behavior contains the PMG action
# instrumentation without spending an episode or leaving a partial result.
if [[ "${STAGE2_GENERATE_ONLY:-0}" == "1" ]]; then
  echo "STAGE2_GENERATE_ONLY=1"
  echo "GENERATED_BEHAVIOR=${behavior_script:-<isaac6-disabled>}"
  echo "GENERATED_BENCHMARK=${headless_script}"
  exit 0
fi

cd "${runtime_dir}"

config_name="$(basename "${config_path}" .yaml)"
result_source="${runtime_dir}/benchmark_results/${run_id}/${config_name}_results_latest.json"
action_trace_source="${runtime_dir}/benchmark_results/${run_id}/action_trace.jsonl"
action_trace_output="${output_root}/${run_id}/action_trace.jsonl"
telemetry_source="${runtime_dir}/logs/${run_id}/carter_ticvla_data"
mkdir -p "${runtime_dir}/benchmark_results/${run_id}"
# The trace is a per-run contract artifact.  Force both runners to this exact
# path and remove only the transient source before launch so a stale trace
# from the same run_id can never satisfy the post-run gate.
export PMG_ACTION_TRACE_PATH="${action_trace_source}"
# Remove the exact run's previous transient and durable trace.  If this
# attempt fails the output directory must not retain an older trace that could
# be mistaken for evidence from the current execution.
rm -f "${result_source}" "${action_trace_source}" "${action_trace_output}"
if [[ -d "${telemetry_source}" ]]; then
  find "${telemetry_source}" -mindepth 1 -delete
fi

gpu_memory_log="${manifest_dir}/${run_id}_gpu_memory.csv"
printf 'epoch,memory_used_mib\n' > "${gpu_memory_log}"
(
  while true; do
    printf '%s,' "$(date +%s)" >> "${gpu_memory_log}"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | head -n 1 >> "${gpu_memory_log}" || true
    sleep 1
  done
) &
monitor_pid=$!
cleanup_monitor() {
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
}
trap cleanup_monitor EXIT

if [[ "${isaac6_compat}" == "1" ]]; then
  compat_runner="${PROJECT_MAIN:-${ticvla_work_root}/project/main}/autodl/scripts/20_run_dynanav_isaac6_compat.py"
  if [[ ! -f "${compat_runner}" ]]; then
    echo "ERROR: Isaac Sim 6 compatibility runner is missing: ${compat_runner}" >&2
    exit 5
  fi
  "${isaac_sim_python}" "${compat_runner}" \
    --benchmark-script "${headless_script}" \
    --config "${config_path}" \
    --run-id "${run_id}" \
    --experience "${experience_path}" \
    "$@"
else
  "${isaac_sim_python}" "${headless_script}" \
    -c "${config_path}" \
    --navigation_method ticvla \
    --run_id "${run_id}" \
    "$@"
fi

cleanup_monitor
trap - EXIT

if [[ ! -f "${result_source}" ]]; then
  echo "ERROR: DynaNav did not produce its result JSON: ${result_source}" >&2
  exit 3
fi

# Treat the post-gate action trace as part of the benchmark contract, not as
# optional telemetry.  Validate every JSONL record and require the complete
# current output schema before copying any result into the durable output
# tree.  M0 is covered by the same schema: its PMG candidate values may be
# null, but the keys and the executed/policy commands must still be present.
# The Isaac launcher prepends its embedded Python 3.12 stdlib to PYTHONPATH.
# Use the host interpreter for this JSONL validator with a clean import path;
# otherwise ``re`` can load the host ``_sre`` extension against Isaac's
# embedded stdlib and fail before the benchmark result is archived.
ACTION_TRACE_SOURCE="${action_trace_source}" env -u PYTHONPATH -u PYTHONHOME -u PYTHONEXE -u LD_PRELOAD python3 - <<'PY'
import json
import os
from pathlib import Path

source = Path(os.environ["ACTION_TRACE_SOURCE"])
required_output_fields = (
    "policy_v_cmd",
    "policy_w_cmd",
    "v_cmd",
    "w_cmd",
    "pmg_override",
    "pmg_override_fraction",
    "pmg_control_source",
    "pmg_gate_scope",
    "pmg_gate_reason",
    "pmg_route_valid",
    "pmg_waypoint_index",
    "pmg_route_alignment_hold",
    "pmg_route_stall_policy_fallback",
    "pmg_semantic_candidate",
    "pmg_control_candidate",
    "pmg_scgp_enabled",
    "pmg_scgp_allow",
    "pmg_scgp_reason",
)

if not source.is_file():
    raise SystemExit(f"ERROR: DynaNav did not produce its action trace: {source}")

output_count = 0

def reject_nonstandard_constant(value: str) -> None:
    # Python's json decoder accepts NaN/Infinity by default, but they are not
    # valid JSON and would make a trace non-portable across evaluators.
    raise ValueError(f"non-standard JSON constant {value}")

with source.open("r", encoding="utf-8") as trace:
    for line_number, raw_line in enumerate(trace, start=1):
        if not raw_line.strip():
            raise SystemExit(
                f"ERROR: action trace contains a blank JSONL record at line {line_number}: {source}"
            )
        try:
            record = json.loads(raw_line, parse_constant=reject_nonstandard_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(
                f"ERROR: action trace JSONL is invalid at line {line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise SystemExit(
                f"ERROR: action trace record at line {line_number} is not a JSON object"
            )
        if record.get("kind") != "output":
            continue
        output_count += 1
        missing = [field for field in required_output_fields if field not in record]
        if missing:
            raise SystemExit(
                "ERROR: action trace output at line "
                f"{line_number} is missing current PMG control fields: {', '.join(missing)}"
            )

if output_count == 0:
    raise SystemExit(f"ERROR: action trace contains no kind=output records: {source}")

print(f"ACTION_TRACE_VALIDATED={source} outputs={output_count}")
PY

cp "${action_trace_source}" "${action_trace_output}"
if ! cmp -s "${action_trace_source}" "${action_trace_output}"; then
  echo "ERROR: archived action trace differs from its runtime source" >&2
  exit 7
fi

cp "${result_source}" "${output_root}/${run_id}/benchmark_metrics.json"
artifact_driver="${PROJECT_MAIN:-${ticvla_work_root}/project/main}/autodl/scripts/16_archive_dynanav_artifacts.py"
telemetry_output="${output_root}/${run_id}/telemetry"
if [[ ! -f "${artifact_driver}" ]]; then
  echo "ERROR: DynaNav artifact driver is missing: ${artifact_driver}" >&2
  exit 4
fi
"${isaac_sim_python}" "${artifact_driver}" \
  --telemetry-dir "${telemetry_source}" \
  --output-dir "${telemetry_output}"
echo "DYNANAV_RESULT=${output_root}/${run_id}/benchmark_metrics.json"
echo "DYNANAV_ACTION_TRACE=${action_trace_output}"
echo "GPU_MEMORY_LOG=${gpu_memory_log}"
echo "DYNANAV_ARTIFACTS=${telemetry_output}/artifacts.json"

# The result, telemetry archive, PMG metrics, and GPU manifest are already
# outside runtime_dir. Remove only this run's transient Kit workspace for
# long matrices so later methods do not inherit stale renderer caches.
if [[ "${run_id}" == stage2_official_v5_fair_* || "${run_id}" == stage2_official_v11_* ]]; then
  find "${runtime_dir}" -mindepth 1 -delete 2>/dev/null || true
fi
