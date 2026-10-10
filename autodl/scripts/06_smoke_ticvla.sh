#!/usr/bin/env bash
set -euo pipefail

ticvla_work_root="${TICVLA_WORK_ROOT:-/root/autodl-tmp/ccfb-ticvla}"
ticvla_repo_dir="${TICVLA_REPO_DIR:-${ticvla_work_root}/src/TIC-VLA}"
ticvla_env_name="${TICVLA_ENV_NAME:-tic-vla}"
ticvla_data_root="${TICVLA_DATA_ROOT:-${ticvla_work_root}/data}"
ticvla_model_root="${TICVLA_BASE_MODEL_PATH:-${ticvla_work_root}/models/InternVL3-1B}"
ticvla_output_root="${TICVLA_OUTPUT_DIR:-${ticvla_work_root}/outputs}"

if [[ ! -d "${ticvla_repo_dir}" ]]; then
  echo "ERROR: TIC-VLA repository is missing. Run 05_setup_ticvla.sh first." >&2
  exit 2
fi

eval "$(conda shell.bash hook)"
export TICVLA_DATA_ROOT="${ticvla_data_root}"
export TICVLA_BASE_MODEL_PATH="${ticvla_model_root}"
export TICVLA_OUTPUT_DIR="${ticvla_output_root}"

cd "${ticvla_repo_dir}"

echo "=== training-stack probe ==="
probe_driver="${PROJECT_MAIN:-${ticvla_work_root}/project/main}/autodl/scripts/13_probe_ticvla_env.py"
if [[ ! -f "${probe_driver}" ]]; then
  echo "ERROR: TIC-VLA environment probe is missing: ${probe_driver}" >&2
  exit 4
fi
conda run --no-capture-output -n "${ticvla_env_name}" python "${probe_driver}"

if [[ ! -f "${ticvla_model_root}/config.json" ]]; then
  echo "WARN: InternVL3-1B config is missing at ${ticvla_model_root}."
  echo "Retrieve InternVL3-1B, the released checkpoint, and DynaNav subset via resource_acquisition_guide.md before model evaluation."
fi

if [[ -n "${ISAAC_SIM_ROOT:-}" && -x "${ISAAC_SIM_ROOT}/python.sh" ]]; then
  echo "=== Isaac Sim probe ==="
  "${ISAAC_SIM_ROOT}/python.sh" -c 'from isaacsim import SimulationApp; print("isaac_sim_import ok")'
else
  echo "WARN: ISAAC_SIM_ROOT/python.sh is unavailable; simulation probe skipped."
fi

if [[ "${TICVLA_RUN_OPEN_LOOP:-0}" == "1" ]]; then
  if [[ -z "${TICVLA_CHECKPOINT_PATH:-}" || ! -f "${TICVLA_CHECKPOINT_PATH}" ]]; then
    echo "ERROR: Set TICVLA_CHECKPOINT_PATH to the released/trained checkpoint." >&2
    exit 3
  fi
  export TICVLA_TEST_DATA_DIR="${TICVLA_TEST_DATA_DIR:-${ticvla_data_root}/DynaNav/DynaNav_json}"
  export TICVLA_TEST_OUTPUT_DIR="${TICVLA_TEST_OUTPUT_DIR:-${ticvla_output_root}/open_loop_eval}"
  open_loop_driver="${PROJECT_MAIN:-${ticvla_work_root}/project/main}/autodl/scripts/10_run_open_loop_baseline.py"
  if [[ ! -f "${open_loop_driver}" ]]; then
    echo "ERROR: Stage 1 open-loop driver is missing: ${open_loop_driver}" >&2
    exit 4
  fi
  conda run --no-capture-output -n "${ticvla_env_name}" \
    python "${open_loop_driver}"
else
  echo "Open-loop evaluation skipped. Set TICVLA_RUN_OPEN_LOOP=1 and TICVLA_CHECKPOINT_PATH when assets are ready."
fi

echo "=== smoke probe complete ==="
