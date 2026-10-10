#!/usr/bin/env bash
set -euo pipefail

root="${STAGE2_RUN_ROOT:-/autodl-fs/data/ccfb-ticvla/stage2_office_run_20261003}"
project="${PROJECT_MAIN:-/autodl-fs/data/10_PROJECT/ccfb-ticvla/project_gate_latch/main}"
prefix="stage2_85_m1_only_compact_20261004"
log_dir="${root}/logs/stage2_official_compat"
mkdir -p "${log_dir}" "${root}/outputs/stage2_official_compat"

export PROJECT_MAIN="${project}"
export STAGE2_RUN_PREFIX="${prefix}"
export STAGE2_PROGRESS_PATH="${root}/outputs/stage2_official_compat/progress_${prefix}.json"
export PMG_CONTROL_POLICY=terminal
export PMG_TERMINAL_ENTER_DISTANCE=11
export PMG_TERMINAL_EXIT_DISTANCE=12
export PMG_ROUTE_ROBOT_RADIUS_M=0.45
export PMG_ROUTE_AXIS_CLEARANCE_X_M=0.50
export PMG_PERCEPTION_SOURCE=isaac_usd_oracle_rgbd
export PMG_SIMPLE_ORACLE_REQUIRE_STABLE=1
export PMG_SIMPLE_ORACLE_STABILITY_MIN_SCORE=0.80
unset STAGE2_STRICT_PARITY

exec 9>"${root}/outputs/stage2_official_compat/single_gpu.lock"
flock -n 9 || { printf 'Another Stage 2 GPU job holds the lock\n' >&2; exit 1; }

log="${log_dir}/${prefix}.log"
{
  printf 'Started: '
  date -u +%FT%TZ
  printf 'Method: M1 only; strict paired action replay disabled\n'
  sha256sum \
    "${project}/scripts/pmg_nav/05_run_official_matrix.py" \
    "${project}/pmg_nav/runtime.py" \
    "${project}/pmg_nav/planner.py" \
    "${project}/autodl/scripts/20_run_dynanav_isaac6_compat.py"
} >>"${log}"

cd "${project}"
set +e
"${root}/isaacsim-6.0.1/python.sh" scripts/pmg_nav/05_run_official_matrix.py \
  --root "${root}" \
  --methods M1 \
  --m1-variant simple_oracle \
  --graph-mode compact \
  --max-graph-nodes 8 \
  --max-edges-per-anchor 4 \
  --min-free-gib 60 \
  --allow-full >>"${log}" 2>&1
status=$?
set -e
{
  printf 'Finished: '
  date -u +%FT%TZ
  printf 'Exit status: %s\n' "${status}"
} >>"${log}"
exit "${status}"
