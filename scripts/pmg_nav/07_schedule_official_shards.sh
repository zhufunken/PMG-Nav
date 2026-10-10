#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

root="${TICVLA_WORK_ROOT:-/root/autodl-tmp/ccfb-ticvla}"
project="${root}/project/main"
sim="${root}/isaacsim-6.0.1/python.sh"
progress_dir="${root}/outputs/stage2_official_compat"
gate_path="${STAGE2_GATE_PATH:-${project}/outputs/stage2/subset_gate.json}"

count_method() {
  local method="$1"
  local total=0
  local dir
  for dir in "${root}/outputs/dynanav_closed_loop"/stage2_official_${method}_e???; do
    if [[ -f "${dir}/benchmark_metrics.json" ]]; then
      total=$((total + 1))
    fi
  done
  printf '%s\n' "${total}"
}

running() {
  screen -ls 2>/dev/null | grep -q "official_stage2_$1" || pgrep -f -- "--methods ${1^^}" >/dev/null 2>&1
}

start_method() {
  local method="$1"
  local progress="${progress_dir}/progress_${method}_auto.json"
  local session="official_stage2_${method}"
  if running "${method}"; then
    return
  fi
  screen -dmS "${session}" bash -lc "export STAGE2_PROGRESS_PATH='${progress}'; cd '${project}'; exec '${sim}' scripts/pmg_nav/05_run_official_matrix.py --methods '${method^^}' --full-after-gate '${gate_path}'"
  echo "started ${method}"
}

while :; do
  m0="$(count_method m0)"
  m1="$(count_method m1)"
  m2="$(count_method m2)"
  m3="$(count_method m3)"
  echo "official shard counts: M0=${m0} M1=${m1} M2=${m2} M3=${m3}"

  if (( m1 >= 85 )); then start_method m3; fi
  if (( m0 >= 85 )); then start_method m2; fi

  if (( m0 >= 85 && m1 >= 85 && m2 >= 85 && m3 >= 85 )); then
    "${sim}" "${project}/scripts/pmg_nav/06_finalize_official_matrix.py" \
      > "${progress_dir}/finalize.log" 2>&1 || true
    echo "official matrix finalized"
    break
  fi
  sleep 300
done
