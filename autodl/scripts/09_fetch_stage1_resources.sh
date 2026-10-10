#!/usr/bin/env bash
set -euo pipefail

ticvla_work_root="${TICVLA_WORK_ROOT:-/root/autodl-tmp/ccfb-ticvla}"
ticvla_env_name="${TICVLA_ENV_NAME:-tic-vla}"
ticvla_data_root="${TICVLA_DATA_ROOT:-${ticvla_work_root}/data/TIC-VLA}"
ticvla_model_root="${TICVLA_BASE_MODEL_PATH:-${ticvla_work_root}/models/InternVL3-1B}"
ticvla_checkpoint_link="${TICVLA_CHECKPOINT_PATH:-${ticvla_work_root}/checkpoints/ticvla.ckpt}"
isaac_installer_dir="${ticvla_work_root}/isaacsim-installers"
isaac_archive="${isaac_installer_dir}/isaac-sim-standalone-5.0.0-linux-x86_64.zip"
manifest_dir="${ticvla_work_root}/manifests"

export HF_HOME="${HF_HOME:-${ticvla_work_root}/cache/huggingface}"
export HF_HUB_DISABLE_XET="${TICVLA_HF_DISABLE_XET:-0}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TICVLA_DATA_ROOT="${ticvla_data_root}"
export TICVLA_BASE_MODEL_PATH="${ticvla_model_root}"

mkdir -p \
  "${ticvla_data_root}" \
  "${ticvla_model_root}" \
  "$(dirname "${ticvla_checkpoint_link}")" \
  "${isaac_installer_dir}" \
  "${manifest_dir}" \
  "${HF_HOME}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is unavailable." >&2
  exit 2
fi

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -Fxq "${ticvla_env_name}"; then
  echo "ERROR: conda environment ${ticvla_env_name} is unavailable." >&2
  exit 3
fi

resource_driver="${PROJECT_MAIN:-${ticvla_work_root}/project/main}/autodl/scripts/14_fetch_stage1_hf.py"
validation_driver="${PROJECT_MAIN:-${ticvla_work_root}/project/main}/autodl/scripts/15_validate_stage1_resources.py"
for driver in "${resource_driver}" "${validation_driver}"; do
  if [[ ! -f "${driver}" ]]; then
    echo "ERROR: Stage 1 resource driver is missing: ${driver}" >&2
    exit 4
  fi
done
conda run --no-capture-output -n "${ticvla_env_name}" python "${resource_driver}"

dynanav_root="${ticvla_data_root}/DynaNav"
checkpoint_source="${ticvla_data_root}/TIC-VLA-model.ckpt"
dynanav_data_zip="${dynanav_root}/DynaNav_data.zip"
dynanav_data_bytes=77480082717
dynanav_data_url="${HF_ENDPOINT%/}/datasets/handsomeYun/TIC-VLA/resolve/main/DynaNav/DynaNav_data.zip?download=true"

mkdir -p "${dynanav_root}"
if [[ ! -f "${dynanav_data_zip}" ]] \
  || [[ "$(stat -c '%s' "${dynanav_data_zip}")" != "${dynanav_data_bytes}" ]]; then
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x 8 -s 8 -k 1M --file-allocation=none \
      -d "${dynanav_root}" -o "$(basename "${dynanav_data_zip}")" \
      "${dynanav_data_url}"
  else
    curl -fL --retry 5 --retry-all-errors -C - \
      -o "${dynanav_data_zip}" "${dynanav_data_url}"
  fi
fi
if [[ "$(stat -c '%s' "${dynanav_data_zip}")" != "${dynanav_data_bytes}" ]]; then
  echo "ERROR: DynaNav_data.zip has an unexpected size." >&2
  exit 4
fi

for required_file in \
  "${ticvla_model_root}/config.json" \
  "${checkpoint_source}" \
  "${dynanav_data_zip}" \
  "${dynanav_root}/DynaNav_json.zip"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: required download is missing: ${required_file}" >&2
    exit 4
  fi
done

unzip -n "${dynanav_root}/DynaNav_data.zip" -d "${dynanav_root}"
unzip -n "${dynanav_root}/DynaNav_json.zip" -d "${dynanav_root}"
test -d "${dynanav_root}/DynaNav_data"
test -d "${dynanav_root}/DynaNav_json"
ln -sfn "${checkpoint_source}" "${ticvla_checkpoint_link}"

validation_manifest="${manifest_dir}/task5_resource_validation.json"
TICVLA_VALIDATION_MANIFEST="${validation_manifest}" \
  conda run --no-capture-output -n "${ticvla_env_name}" python "${validation_driver}"

if [[ "${TICVLA_DOWNLOAD_ISAAC_SIM:-1}" == "1" && ! -f "${isaac_archive}" ]]; then
  isaac_url="https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.0.0-linux-x86_64.zip"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x 8 -s 8 -d "${isaac_installer_dir}" \
      -o "$(basename "${isaac_archive}")" "${isaac_url}"
  else
    curl -fL --retry 5 --retry-all-errors -C - \
      -o "${isaac_archive}" "${isaac_url}"
  fi
fi

resources_tsv="${manifest_dir}/stage1_resources.tsv"
{
  printf 'resource\tbytes\tpath\tsource\n'
  printf 'InternVL3-1B\t%s\t%s\t%s\n' "$(du -sb "${ticvla_model_root}" | awk '{print $1}')" "${ticvla_model_root}" "${HF_ENDPOINT}"
  printf 'TIC-VLA-checkpoint\t%s\t%s\t%s\n' "$(stat -c '%s' "${checkpoint_source}")" "${checkpoint_source}" "${HF_ENDPOINT}"
  printf 'DynaNav-data-zip\t%s\t%s\t%s\n' "$(stat -c '%s' "${dynanav_root}/DynaNav_data.zip")" "${dynanav_root}/DynaNav_data.zip" "${HF_ENDPOINT}"
  printf 'DynaNav-json-zip\t%s\t%s\t%s\n' "$(stat -c '%s' "${dynanav_root}/DynaNav_json.zip")" "${dynanav_root}/DynaNav_json.zip" "${HF_ENDPOINT}"
  printf 'DynaNav-extracted\t%s\t%s\t%s\n' "$(du -sb "${dynanav_root}" | awk '{print $1}')" "${dynanav_root}" "${HF_ENDPOINT}"
  if [[ -f "${isaac_archive}" ]]; then
    printf 'Isaac-Sim-5.0.0\t%s\t%s\t%s\n' "$(stat -c '%s' "${isaac_archive}")" "${isaac_archive}" 'NVIDIA-official'
  fi
} > "${resources_tsv}"

checksums="${manifest_dir}/stage1_checksums.sha256"
checksum_files=(
  "${ticvla_model_root}/model.safetensors" \
  "${checkpoint_source}" \
  "${dynanav_root}/DynaNav_data.zip" \
  "${dynanav_root}/DynaNav_json.zip"
)
if [[ -f "${isaac_archive}" ]]; then
  checksum_files+=("${isaac_archive}")
fi
sha256sum "${checksum_files[@]}" > "${checksums}"

echo "RESOURCE_MANIFEST=${resources_tsv}"
echo "CHECKSUM_MANIFEST=${checksums}"
echo "VALIDATION_MANIFEST=${validation_manifest}"
df -h "${ticvla_work_root}"
