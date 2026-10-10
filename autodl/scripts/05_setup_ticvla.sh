#!/usr/bin/env bash
set -euo pipefail

ticvla_work_root="${TICVLA_WORK_ROOT:-/root/autodl-tmp/ccfb-ticvla}"
ticvla_repo_dir="${TICVLA_REPO_DIR:-${ticvla_work_root}/src/TIC-VLA}"
ticvla_env_name="${TICVLA_ENV_NAME:-tic-vla}"
ticvla_data_root="${TICVLA_DATA_ROOT:-${ticvla_work_root}/data}"
ticvla_model_root="${TICVLA_BASE_MODEL_PATH:-${ticvla_work_root}/models/InternVL3-1B}"
ticvla_output_root="${TICVLA_OUTPUT_DIR:-${ticvla_work_root}/outputs}"
conda_bootstrap_channel="${CONDA_BOOTSTRAP_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main}"
export PIP_INDEX_URL="${TICVLA_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable." >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required. Select an AutoDL Miniconda/PyTorch image." >&2
  exit 3
fi

eval "$(conda shell.bash hook)"
mkdir -p "${ticvla_work_root}/src" "${ticvla_data_root}" "${ticvla_model_root}" "${ticvla_output_root}"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
echo "PIP_INDEX_URL=${PIP_INDEX_URL}"

if [[ ! -d "${ticvla_repo_dir}/.git" ]]; then
  git clone --depth 1 https://github.com/ucla-mobility/TIC-VLA.git "${ticvla_repo_dir}"
else
  echo "Using existing repository: ${ticvla_repo_dir}"
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "${ticvla_env_name}"; then
  conda create -y -n "${ticvla_env_name}" \
    --override-channels -c "${conda_bootstrap_channel}" \
    python=3.11 pip
else
  echo "Using existing conda environment: ${ticvla_env_name}"
fi

conda run --no-capture-output -n "${ticvla_env_name}" \
  python -m pip install -r "${ticvla_repo_dir}/requirements-train.txt"
conda run -n "${ticvla_env_name}" python -m pip install -e "${ticvla_repo_dir}"

probe_driver="${PROJECT_MAIN:-${ticvla_work_root}/project/main}/autodl/scripts/13_probe_ticvla_env.py"
if [[ ! -f "${probe_driver}" ]]; then
  echo "ERROR: TIC-VLA environment probe is missing: ${probe_driver}" >&2
  exit 6
fi
conda run --no-capture-output -n "${ticvla_env_name}" python "${probe_driver}"

if [[ "${TICVLA_DOWNLOAD_ASSETS:-0}" == "1" ]]; then
  export TICVLA_DATA_ROOT="${ticvla_data_root}"
  export TICVLA_BASE_MODEL_PATH="${ticvla_model_root}"
  conda run -n "${ticvla_env_name}" python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="handsomeYun/TIC-VLA",
    repo_type="dataset",
    local_dir=os.environ["TICVLA_DATA_ROOT"],
)
snapshot_download(
    repo_id="OpenGVLab/InternVL3-1B",
    repo_type="model",
    local_dir=os.environ["TICVLA_BASE_MODEL_PATH"],
)
PY
else
  echo "Asset download skipped. Use resource_acquisition_guide.md for selective checkpoint/DynaNav/model retrieval."
  echo "TICVLA_DOWNLOAD_ASSETS=1 downloads the full ~538 GB dataset and is reserved for full SFT."
fi

if [[ -n "${ISAAC_SIM_ARCHIVE:-}" ]]; then
  if [[ ! -f "${ISAAC_SIM_ARCHIVE}" ]]; then
    echo "ERROR: ISAAC_SIM_ARCHIVE does not exist: ${ISAAC_SIM_ARCHIVE}" >&2
    exit 4
  fi
  isaac_sim_root="${ISAAC_SIM_ROOT:-${ticvla_work_root}/isaacsim}"
  mkdir -p "${isaac_sim_root}"
  unzip -n "${ISAAC_SIM_ARCHIVE}" -d "${isaac_sim_root}"
  if [[ -x "${isaac_sim_root}/post_install.sh" ]]; then
    "${isaac_sim_root}/post_install.sh"
  fi
  if [[ ! -x "${isaac_sim_root}/python.sh" ]]; then
    echo "ERROR: Isaac Sim python.sh is unavailable after extraction." >&2
    exit 5
  fi
  "${isaac_sim_root}/python.sh" -m pip install -e "${ticvla_repo_dir}"
  "${isaac_sim_root}/python.sh" -m pip install -r "${ticvla_repo_dir}/requirements-test.txt"
  echo "Isaac Sim extracted to ${isaac_sim_root}"
else
  echo "Isaac Sim was not installed. Put the official 5.0.0 Linux archive on the data disk and set ISAAC_SIM_ARCHIVE."
fi

echo "=== setup complete ==="
printf 'TICVLA_REPO_DIR=%s\n' "${ticvla_repo_dir}"
printf 'TICVLA_DATA_ROOT=%s\n' "${ticvla_data_root}"
printf 'TICVLA_BASE_MODEL_PATH=%s\n' "${ticvla_model_root}"
printf 'TICVLA_OUTPUT_DIR=%s\n' "${ticvla_output_root}"
