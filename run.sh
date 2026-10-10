#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1 && [[ -x /root/miniconda3/bin/conda ]]; then
  export PATH="/root/miniconda3/bin:${PATH}"
fi

PMG_ROOT="${PMG_ROOT:-/root/autodl-tmp/ccfb-ticvla}"
PROJECT_MAIN="${PROJECT_MAIN:-${PMG_ROOT}/project/main}"
TICVLA_REPO_DIR="${TICVLA_REPO_DIR:-${PMG_ROOT}/src/TIC-VLA}"
TICVLA_BASE_MODEL_PATH="${TICVLA_BASE_MODEL_PATH:-${PMG_ROOT}/models/InternVL3-1B}"
TICVLA_CHECKPOINT_PATH="${TICVLA_CHECKPOINT_PATH:-${PMG_ROOT}/checkpoints/ticvla.ckpt}"
TICVLA_DATA_ROOT="${TICVLA_DATA_ROOT:-${PMG_ROOT}/data/TIC-VLA}"
TICVLA_OUTPUT_DIR="${TICVLA_OUTPUT_DIR:-${PMG_ROOT}/outputs}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-${PMG_ROOT}/isaacsim}"
TICVLA_COMMIT="9fa6f8b66b9e121d5df5df071297bba8e5353ebb"

export PMG_ROOT PROJECT_MAIN TICVLA_REPO_DIR TICVLA_BASE_MODEL_PATH
export TICVLA_CHECKPOINT_PATH TICVLA_DATA_ROOT TICVLA_OUTPUT_DIR ISAAC_SIM_ROOT

mkdir -p \
  "${PMG_ROOT}/logs" \
  "${PMG_ROOT}/manifests" \
  "${PMG_ROOT}/outputs"

run_logged() {
  local task_name="$1"
  local log_file="$2"
  shift 2

  local start end rc
  start="$(date +%s)"
  set +e
  "$@" 2>&1 | tee "${log_file}"
  rc="${PIPESTATUS[0]}"
  set -e
  end="$(date +%s)"

  {
    printf 'task=%s\n' "${task_name}"
    printf 'started_epoch=%s\n' "${start}"
    printf 'finished_epoch=%s\n' "${end}"
    printf 'duration_seconds=%s\n' "$((end - start))"
    printf 'exit_code=%s\n' "${rc}"
  } > "${PMG_ROOT}/manifests/${task_name}_status.env"
  sync
  return "${rc}"
}

task2() {
  cd "${PROJECT_MAIN}/autodl"
  run_logged task2 "${PMG_ROOT}/logs/task2_instance_probe.log" \
    bash scripts/00_probe_instance.sh

  {
    echo '# Task 2 Network Status'
    echo
    printf -- '- Checked: %s\n' "$(date -Is)"
    echo
    echo '```text'
    sed -n '/=== network-readiness ===/,/=== probe-complete ===/p' \
      "${PMG_ROOT}/logs/task2_instance_probe.log"
    echo '```'
  } > "${PMG_ROOT}/manifests/task2_network_status.md"
}

task3() {
  mkdir -p "${PMG_ROOT}/src"
  if [[ -d "${TICVLA_REPO_DIR}" ]]; then
    find "${TICVLA_REPO_DIR}" -type f -name '._*' -delete
  fi
  if [[ -d "${TICVLA_REPO_DIR}/.git" ]] \
    && [[ "$(git -C "${TICVLA_REPO_DIR}" rev-parse HEAD)" == "${TICVLA_COMMIT}" ]] \
    && [[ -f "${TICVLA_REPO_DIR}/ticvla/__init__.py" ]] \
    && [[ -f "${TICVLA_REPO_DIR}/DynaNav/benchmark.py" ]]; then
    run_logged task3 "${PMG_ROOT}/logs/task3_clone_ticvla.log" \
      git -C "${TICVLA_REPO_DIR}" status --short
  elif [[ ! -d "${TICVLA_REPO_DIR}/.git" ]]; then
    run_logged task3 "${PMG_ROOT}/logs/task3_clone_ticvla.log" \
      git clone --filter=blob:none --no-checkout --depth 1 \
        https://github.com/ucla-mobility/TIC-VLA.git \
        "${TICVLA_REPO_DIR}"
    git -C "${TICVLA_REPO_DIR}" fetch --depth 1 origin "${TICVLA_COMMIT}"
    git -C "${TICVLA_REPO_DIR}" sparse-checkout init --no-cone
    git -C "${TICVLA_REPO_DIR}" sparse-checkout set --no-cone \
      '/*' \
      '!/*/' \
      '/ticvla/' \
      '/configs/' \
      '/DynaNav/*.py' \
      '/DynaNav/*.sh' \
      '/DynaNav/*.md' \
      '/DynaNav/configs/' \
      '/DynaNav/behavior/' \
      '/rl/*.py' \
      '/rl/*.md' \
      '/rl/fine_tuning/'
    git -C "${TICVLA_REPO_DIR}" checkout --detach "${TICVLA_COMMIT}"
  else
    echo "ERROR: existing TIC-VLA repository is incomplete or on the wrong commit" >&2
    return 4
  fi
  {
    git -C "${TICVLA_REPO_DIR}" rev-parse HEAD
    git -C "${TICVLA_REPO_DIR}" branch --show-current
    git -C "${TICVLA_REPO_DIR}" status --short
  } > "${PMG_ROOT}/manifests/ticvla_commit.txt"

  test -f "${TICVLA_REPO_DIR}/README.md"
  test -d "${TICVLA_REPO_DIR}/ticvla"
  test -d "${TICVLA_REPO_DIR}/DynaNav"
  test -d "${TICVLA_REPO_DIR}/rl"
}

task4() {
  cd "${PROJECT_MAIN}/autodl"
  TICVLA_DOWNLOAD_ASSETS=0 run_logged task4_setup \
    "${PMG_ROOT}/logs/task4_setup_ticvla.log" \
    bash scripts/05_setup_ticvla.sh

  conda run -n tic-vla python -m pip freeze \
    > "${PMG_ROOT}/manifests/ticvla_pip_freeze.txt"
  run_logged task4_smoke "${PMG_ROOT}/logs/task4_ticvla_smoke.log" \
    bash scripts/06_smoke_ticvla.sh
}

task5() {
  cd "${PROJECT_MAIN}/autodl"
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
    run_logged task5 "${PMG_ROOT}/logs/task5_resource_download.log" \
    bash scripts/09_fetch_stage1_resources.sh
}

task6() {
  cd "${PROJECT_MAIN}/autodl"
  TICVLA_RUN_OPEN_LOOP=1 run_logged task6 \
    "${PMG_ROOT}/logs/task6_open_loop.log" \
    bash scripts/06_smoke_ticvla.sh
}

verify_isaac6_stage1_env() {
  test -x "${ISAAC_SIM_ROOT}/python.sh"
  test -f "${TICVLA_REPO_DIR}/DynaNav/benchmark.py"
  test -f "${TICVLA_BASE_MODEL_PATH}/config.json"
  test -f "${TICVLA_CHECKPOINT_PATH}"
  "${ISAAC_SIM_ROOT}/python.sh" - <<'PY'
import cv2
import pytorch_lightning
import torch
import transformers

if torch.__version__ != "2.8.0+cu128":
    raise RuntimeError(f"expected torch 2.8.0+cu128; found {torch.__version__}")
if torch.version.cuda != "12.8":
    raise RuntimeError(f"RTX 5090 requires the cu128 build; found torch CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access the RTX 5090")
if torch.cuda.get_device_capability(0) != (12, 0):
    raise RuntimeError(f"unexpected GPU capability: {torch.cuda.get_device_capability(0)}")
if "sm_120" not in torch.cuda.get_arch_list():
    raise RuntimeError(f"PyTorch build has no Blackwell kernel target: {torch.cuda.get_arch_list()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"transformers={transformers.__version__}")
print(f"opencv={cv2.__version__}")
print(f"pytorch_lightning={pytorch_lightning.__version__}")
PY
  "${ISAAC_SIM_ROOT}/python.sh" -m pip freeze \
    > "${PMG_ROOT}/manifests/isaac6_pip_freeze.txt"
}

task7() {
  cd "${PROJECT_MAIN}/autodl"
  run_logged task7_vulkan_probe \
    "${PMG_ROOT}/logs/task7_vulkan_probe.log" \
    bash scripts/17_probe_nvidia_vulkan.sh

  if [[ "${TICVLA_ISAAC6_COMPAT:-0}" == "1" || "${ISAAC_SIM_ROOT##*/}" == isaacsim-6.* ]]; then
    run_logged task7_setup \
      "${PMG_ROOT}/logs/task7_isaacsim_setup.log" \
      verify_isaac6_stage1_env
  else
    ISAAC_SIM_ARCHIVE="${ISAAC_SIM_ARCHIVE:-${PMG_ROOT}/isaacsim-installers/isaac-sim-standalone-5.0.0-linux-x86_64.zip}" \
      ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT}" \
      TICVLA_DOWNLOAD_ASSETS=0 run_logged task7_setup \
        "${PMG_ROOT}/logs/task7_isaacsim_setup.log" \
        bash scripts/05_setup_ticvla.sh
  fi

  CUDA_VISIBLE_DEVICES=0 run_logged task7_closed_loop \
    "${PMG_ROOT}/logs/task7_dynanav_closed_loop.log" \
    bash scripts/08_run_dynanav_baseline.sh \
      "${PROJECT_MAIN}/autodl/configs/dynanav_stage1_smoke.yaml"

  python3 scripts/11_collect_dynanav_metrics.py \
    --result "${PMG_ROOT}/outputs/dynanav_closed_loop/task7/benchmark_metrics.json" \
    --log "${PMG_ROOT}/logs/task7_dynanav_closed_loop.log" \
    --gpu-log "${PMG_ROOT}/manifests/task7_gpu_memory.csv" \
    --output "${PMG_ROOT}/manifests/task7_closed_loop_metrics.json"
}

task8() {
  cd "${PROJECT_MAIN}/autodl"
  mkdir -p "${PMG_ROOT}/outputs/stage1_final_recheck"

  TICVLA_RUN_OPEN_LOOP=1 \
    TICVLA_OPEN_LOOP_INDICES=0 \
    TICVLA_TEST_OUTPUT_DIR="${PMG_ROOT}/outputs/stage1_final_recheck/open_loop" \
    TICVLA_OPEN_LOOP_METRICS="${PMG_ROOT}/manifests/task8_open_loop_metrics.json" \
    run_logged task8_open_loop \
      "${PMG_ROOT}/logs/task8_open_loop_recheck.log" \
      bash scripts/06_smoke_ticvla.sh

  TICVLA_DYNANAV_RUN_ID=task8 \
    CUDA_VISIBLE_DEVICES=0 \
    run_logged task8_closed_loop \
      "${PMG_ROOT}/logs/task8_dynanav_recheck.log" \
      bash scripts/08_run_dynanav_baseline.sh \
        "${PROJECT_MAIN}/autodl/configs/dynanav_stage1_smoke.yaml"

  python3 scripts/11_collect_dynanav_metrics.py \
    --result "${PMG_ROOT}/outputs/dynanav_closed_loop/task8/benchmark_metrics.json" \
    --log "${PMG_ROOT}/logs/task8_dynanav_recheck.log" \
    --gpu-log "${PMG_ROOT}/manifests/task8_gpu_memory.csv" \
    --output "${PMG_ROOT}/manifests/task8_closed_loop_metrics.json"

  mkdir -p "${PMG_ROOT}/outputs/stage1_final_recheck/closed_loop"
  cp "${PMG_ROOT}/outputs/dynanav_closed_loop/task8/benchmark_metrics.json" \
    "${PMG_ROOT}/outputs/stage1_final_recheck/closed_loop/benchmark_metrics.json"

  run_logged task8_finalize \
    "${PMG_ROOT}/logs/task8_finalize.log" \
    python3 scripts/12_finalize_stage1.py
}

case "${1:-}" in
  2) task2 ;;
  3) task3 ;;
  4) task4 ;;
  5) task5 ;;
  6) task6 ;;
  7) task7 ;;
  8) task8 ;;
  *) echo "Usage: bash run.sh {2|3|4|5|6|7|8}" >&2; exit 2 ;;
esac
