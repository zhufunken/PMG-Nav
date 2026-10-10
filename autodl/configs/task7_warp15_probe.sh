#!/usr/bin/env bash
set -euo pipefail

root="${TICVLA_WORK_ROOT:-/root/autodl-tmp/ccfb-ticvla}"
log="${root}/logs/task7_warp15_probe.log"
status="${root}/logs/task7_warp15_probe.status"
pid_file="${root}/logs/task7_warp15_probe.pid"
warp_target="${root}/runtime/warp-1.15.0"

if [[ "${1:-}" != "worker" ]]; then
  nohup bash "$0" worker >"${log}" 2>&1 </dev/null &
  printf '%s\n' "$!" >"${pid_file}"
  echo "TASK7_WARP15_PROBE_STARTED $!"
  exit 0
fi

started_at="$(date +%s)"
trap 'rc=$?; printf "exit_code=%s\nelapsed_seconds=%s\nfinished_at=%s\n" "$rc" "$(( $(date +%s) - started_at ))" "$(date -Is)" >"${status}"; exit "$rc"' EXIT

mkdir -p "${warp_target}"
if [[ ! -f "${warp_target}/warp/__init__.py" ]]; then
  "${root}/isaacsim/python.sh" -m pip install \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --target "${warp_target}" \
    --no-cache-dir \
    warp-lang==1.15.0
fi

system_lib_dir="${root}/runtime/isaac-system-libs/root/usr/lib/x86_64-linux-gnu"
glu_dir="${root}/runtime/libglu1-mesa/root/usr/lib/x86_64-linux-gnu"
test -e "${system_lib_dir}/libXt.so.6"
test -e "${glu_dir}/libGLU.so.1"

export PYTHONPATH="${warp_target}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${system_lib_dir}:${glu_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json
export OMNI_KIT_ACCEPT_EULA=YES
unset CUDA_VISIBLE_DEVICES

echo "=== Warp 1.15 probe: $(date -Is) ==="
timeout 600 "${root}/isaacsim/python.sh" - <<'PY'
import warp

print("PRELOADED_WARP_VERSION", warp.__version__, flush=True)
print("PRELOADED_WARP_PATH", warp.__file__, flush=True)

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
print("ISAAC_SIM_HEADLESS_READY", flush=True)
app.update()
app.close()
print("ISAAC_SIM_HEADLESS_CLOSED", flush=True)
PY

echo "TASK7_WARP15_PROBE_PASSED"
