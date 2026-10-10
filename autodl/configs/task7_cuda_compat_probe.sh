#!/usr/bin/env bash
set -euo pipefail

root="${TICVLA_WORK_ROOT:-/root/autodl-tmp/ccfb-ticvla}"
log="${root}/logs/task7_cuda12_compat_probe.log"
status="${root}/logs/task7_cuda12_compat_probe.status"
pid_file="${root}/logs/task7_cuda12_compat_probe.pid"

if [[ "${1:-}" != "worker" ]]; then
  nohup bash "$0" worker >"${log}" 2>&1 </dev/null &
  printf '%s\n' "$!" >"${pid_file}"
  echo "TASK7_CUDA12_COMPAT_PROBE_STARTED $!"
  exit 0
fi

started_at="$(date +%s)"
trap 'rc=$?; printf "exit_code=%s\nelapsed_seconds=%s\nfinished_at=%s\n" "$rc" "$(( $(date +%s) - started_at ))" "$(date -Is)" >"${status}"; exit "$rc"' EXIT

compat_dir="/usr/local/cuda-12.8/compat"
system_lib_dir="${root}/runtime/isaac-system-libs/root/usr/lib/x86_64-linux-gnu"
glu_dir="${root}/runtime/libglu1-mesa/root/usr/lib/x86_64-linux-gnu"

test -e "${compat_dir}/libcuda.so.1"
test -e "${system_lib_dir}/libXt.so.6"
test -e "${glu_dir}/libGLU.so.1"

export LD_LIBRARY_PATH="${compat_dir}:${system_lib_dir}:${glu_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json
export OMNI_KIT_ACCEPT_EULA=YES
unset CUDA_VISIBLE_DEVICES

echo "=== CUDA 12.8 compatibility probe: $(date -Is) ==="
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
ldd "${root}/isaacsim/kit/kernel/plugins/libcarb.so" | grep -E 'libcuda|not found' || true

timeout 300 "${root}/isaacsim/python.sh" - <<'PY'
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
print("ISAAC_SIM_HEADLESS_READY", flush=True)
app.update()
app.close()
print("ISAAC_SIM_HEADLESS_CLOSED", flush=True)
PY

echo "TASK7_CUDA12_COMPAT_PROBE_PASSED"
