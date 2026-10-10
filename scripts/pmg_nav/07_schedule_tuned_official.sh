#!/usr/bin/env bash
set -euo pipefail

root="${TICVLA_WORK_ROOT:-/root/autodl-tmp/ccfb-ticvla}"
project="${root}/project/main"
sim="${root}/isaacsim-6.0.1/python.sh"
progress_dir="${root}/outputs/stage2_official_compat"
prefix="${STAGE2_RUN_PREFIX:-stage2_official_tuned}"
progress_name="${STAGE2_PROGRESS_NAME:-tuned}"
baseline_prefix="${STAGE2_M0_PREFIX:-stage2_official_v5_fair}"
baseline_progress_name="${STAGE2_M0_PROGRESS_NAME:-v5_fair}"
gate_path="${STAGE2_GATE_PATH:-${project}/outputs/stage2/subset_gate.json}"

valid_count() {
  ROOT="$root" METHOD="$1" PREFIX="$prefix" PROGRESS_NAME="$progress_name" \
    BASELINE_PREFIX="$baseline_prefix" BASELINE_PROGRESS_NAME="$baseline_progress_name" \
    SKIP_M0="${SKIP_M0:-0}" python3 - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ["ROOT"])
method = os.environ["METHOD"].lower()
prefix = os.environ["PREFIX"]
progress_name = os.environ.get("PROGRESS_NAME", "tuned")
baseline_prefix = os.environ.get("BASELINE_PREFIX", "stage2_official_v5_fair")
baseline_progress_name = os.environ.get("BASELINE_PROGRESS_NAME", "v5_fair")
skip_m0 = os.environ.get("SKIP_M0", "0") == "1"
selected_prefix = baseline_prefix if method == "m0" and skip_m0 else prefix
selected_progress_name = baseline_progress_name if method == "m0" and skip_m0 else progress_name
progress_path = root / "outputs/stage2_official_compat" / f"progress_{selected_progress_name}_{method}.json"
try:
    progress = json.loads(progress_path.read_text(encoding="utf-8")).get("runs", {})
except (OSError, json.JSONDecodeError):
    progress = {}
count = 0
for index in range(1, 86):
    run_id = f"{selected_prefix}_{method}_e{index:03d}"
    # Failed/OOM runs can leave partial JSON files; only a recorded passed
    # process with complete telemetry may advance the scheduler.
    if progress.get(run_id, {}).get("status") not in {"passed", "cached"}:
        continue
    result = root / "outputs/dynanav_closed_loop" / run_id / "benchmark_metrics.json"
    artifacts = root / "outputs/dynanav_closed_loop" / run_id / "telemetry/artifacts.json"
    try:
        episode = json.loads(result.read_text(encoding="utf-8"))["episodes"][0]
        artifact_payload = json.loads(artifacts.read_text(encoding="utf-8"))
        valid = (episode.get("episode") == f"episode_{index}" and not episode.get("error")
                 and int(episode.get("runtime_num_people", -1)) == 0
                 and int(artifact_payload.get("rgb_frames", 0) or 0) > 0
                 and int(artifact_payload.get("trajectory_points", 0) or 0) > 0)
        if method != "m0":
            pmg = root / "runtime/dynanav/pmg_metrics" / run_id / f"episode_{index}.json"
            payload = json.loads(pmg.read_text(encoding="utf-8"))
            valid = valid and payload.get("observations", 0) >= 2 and payload.get("graph_nodes", 0) >= 1
        count += int(valid)
    except Exception:
        pass
print(count)
PY
}

active() {
  # A detached screen can remain as an empty socket after the runner exits.
  # Only a live matrix runner should block the next retry/start decision.
  local method="${1^^}"
  pgrep -f -- "05_run_official_matrix.py --methods ${method}( |$)" >/dev/null 2>&1
}

start_method() {
  local method="$1"
  active "$method" && return 0
  screen -dmS "official_tuned_${method}" bash -lc \
    "export STAGE2_RUN_PREFIX='${prefix}'; export STAGE2_PROGRESS_PATH='${progress_dir}/progress_${progress_name}_${method}.json'; export PMG_FRAME_STRIDE='${PMG_FRAME_STRIDE:-15}'; export PMG_RECOVERY_MIN_SCORE='${PMG_RECOVERY_MIN_SCORE:-0.92}'; export PMG_RECOVERY_MAX_DISTANCE='${PMG_RECOVERY_MAX_DISTANCE:-4.0}'; export PMG_RECOVERY_STALL_SPEED='${PMG_RECOVERY_STALL_SPEED:-0.18}'; export PMG_RECOVERY_MIN_HEADING='${PMG_RECOVERY_MIN_HEADING:-1.05}'; export PMG_RECOVERY_BLEND='${PMG_RECOVERY_BLEND:-0.35}'; export TICVLA_LOG_INTERVAL_FRAMES='30'; export TICVLA_SKIP_SCENE_LOAD_WAIT='1'; cd '${project}'; exec '${sim}' scripts/pmg_nav/05_run_official_matrix.py --methods '${method^^}' --full-after-gate '${gate_path}'"
  echo "started ${method}"
}

while :; do
  m0="$(valid_count m0)"
  m1="$(valid_count m1)"
  m2="$(valid_count m2)"
  m3="$(valid_count m3)"
  echo "tuned counts M0=${m0} M1=${m1} M2=${m2} M3=${m3}"
  # Re-enter failed/incomplete baseline shards after a runner exits.  The
  # runner itself scans all 85 episodes and retries entries whose previous
  # process status was not passed.
  # The first pass may have two legacy shards already running.  Once those
  # exit, retry baseline shards one at a time to avoid sharing the VLA/RTX
  # memory pool and reproducing the PhysX/renderer failures.
  if (( m0 < 85 )) && ! active m1 && [[ "${SKIP_M0:-0}" != "1" ]]; then
    start_method m0
  fi
  if (( m0 >= 85 && m1 < 85 )) && ! active m0; then
    start_method m1
  fi
  if (( m0 >= 85 && m1 >= 85 && m2 < 85 )); then
    start_method m2
  fi
  # PMG methods load the same VLA checkpoint. Keep M3 behind M2 so only one
  # PMG model is resident alongside the M0/M1 baseline shards; this avoids
  # GPU OOM while preserving cached valid episodes.
  if (( m0 >= 85 && m1 >= 85 && m2 >= 85 && m3 < 85 )); then
    start_method m3
  fi
  if (( m0 >= 85 && m1 >= 85 && m2 >= 85 && m3 >= 85 )); then
    export STAGE2_RUN_PREFIX="${prefix}"
    STAGE2_M0_PREFIX="${baseline_prefix}" \
    "${sim}" "${project}/scripts/pmg_nav/06_finalize_official_matrix.py" \
      > "${progress_dir}/${prefix}_finalize.log" 2>&1
    echo "tuned matrix finalized"
    break
  fi
  sleep 300
done
