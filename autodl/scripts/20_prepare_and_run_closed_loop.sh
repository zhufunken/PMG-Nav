#!/usr/bin/env bash
set -euo pipefail

work_root="${PMG_ROOT:-/root/autodl-tmp/ccfb-ticvla}"
project_main="${PROJECT_MAIN:-${work_root}/project/main}"
log_dir="${work_root}/logs"
log_file="${log_dir}/task7_restore_and_closed_loop.log"
status_file="${log_dir}/task7_restore_and_closed_loop.status"
pid_file="${log_dir}/task7_restore_and_closed_loop.pid"

mkdir -p "${log_dir}" "${work_root}/cache/huggingface"

if [[ "${1:-}" != "--worker" ]]; then
  nohup bash "$0" --worker >>"${log_file}" 2>&1 </dev/null &
  worker_pid=$!
  printf '%s\n' "${worker_pid}" >"${pid_file}"
  echo "Task 7 restore and closed-loop run started: PID ${worker_pid}"
  echo "Log: ${log_file}"
  exit 0
fi

started_at="$(date +%s)"
trap 'rc=$?; elapsed=$(( $(date +%s) - started_at )); printf "exit_code=%s\nelapsed_seconds=%s\nfinished_at=%s\n" "${rc}" "${elapsed}" "$(date -Is)" >"${status_file}"; exit "${rc}"' EXIT

export PMG_ROOT="${work_root}"
export PROJECT_MAIN="${project_main}"
export TICVLA_REPO_DIR="${work_root}/src/TIC-VLA"
export TICVLA_BASE_MODEL_PATH="${work_root}/models/InternVL3-1B"
export TICVLA_CHECKPOINT_PATH="${work_root}/checkpoints/ticvla.ckpt"
export TICVLA_DATA_ROOT="${work_root}/data/TIC-VLA"
export TICVLA_OUTPUT_DIR="${work_root}/outputs"
export ISAAC_SIM_ROOT="${work_root}/isaacsim"
export HF_HOME="${work_root}/cache/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_XET_HIGH_PERFORMANCE=1
export PIP_CACHE_DIR="${work_root}/cache/pip"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json

echo "=== Task 7 restore started: $(date -Is) ==="
bash "${project_main}/run.sh" 3
bash "${project_main}/run.sh" 4

eval "$(conda shell.bash hook)"
conda run --no-capture-output -n tic-vla \
  python "${project_main}/autodl/scripts/19_fetch_closed_loop_resources.py"

bash "${project_main}/run.sh" 7
echo "=== Task 7 closed-loop passed: $(date -Is) ==="
sync
