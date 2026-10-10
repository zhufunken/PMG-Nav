#!/usr/bin/env bash
set -euo pipefail

work_root="${TICVLA_WORK_ROOT:-/root/autodl-tmp/ccfb-ticvla}"
installer_dir="${work_root}/isaacsim-installers"
isaac_root="${ISAAC_SIM_ROOT:-${work_root}/isaacsim}"
archive="${installer_dir}/isaac-sim-standalone-5.0.0-linux-x86_64.zip"
log_dir="${work_root}/logs"
log_file="${log_dir}/task7_isaacsim_headless_probe.log"
status_file="${log_dir}/task7_isaacsim_headless_probe.status"
pid_file="${log_dir}/task7_isaacsim_headless_probe.pid"

mkdir -p "${installer_dir}" "${isaac_root}" "${log_dir}"

if [[ "${1:-}" != "--worker" ]]; then
  nohup bash "$0" --worker >>"${log_file}" 2>&1 </dev/null &
  worker_pid=$!
  printf '%s\n' "${worker_pid}" >"${pid_file}"
  echo "Isaac Sim headless probe started in background: PID ${worker_pid}"
  echo "Log: ${log_file}"
  exit 0
fi

started_at="$(date +%s)"
trap 'rc=$?; elapsed=$(( $(date +%s) - started_at )); printf "exit_code=%s\nelapsed_seconds=%s\nfinished_at=%s\n" "${rc}" "${elapsed}" "$(date -Is)" >"${status_file}"; exit "${rc}"' EXIT

echo "=== Isaac Sim headless probe started: $(date -Is) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
df -h "${work_root}"

if [[ ! -f "${archive}" ]]; then
  isaac_url="https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.0.0-linux-x86_64.zip"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x 8 -s 8 --file-allocation=none \
      -d "${installer_dir}" -o "$(basename "${archive}")" "${isaac_url}"
  else
    curl -fL --retry 5 --retry-all-errors -C - \
      -o "${archive}" "${isaac_url}"
  fi
fi

unzip -n "${archive}" -d "${isaac_root}"
if [[ -x "${isaac_root}/post_install.sh" ]]; then
  "${isaac_root}/post_install.sh"
fi
test -x "${isaac_root}/python.sh"

export VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json
export CUDA_VISIBLE_DEVICES=0
"${isaac_root}/python.sh" - <<'PY'
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
print("ISAAC_SIM_HEADLESS_READY")
app.update()
app.close()
PY

echo "=== Isaac Sim headless probe passed: $(date -Is) ==="
