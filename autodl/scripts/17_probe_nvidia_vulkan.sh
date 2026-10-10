#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${script_dir}/16_configure_autodl_vulkan.sh"

for device in /dev/nvidia0 /dev/nvidiactl; do
  if [[ ! -e "${device}" ]]; then
    echo "WARNING: standard NVIDIA device node is missing: ${device}" >&2
  fi
done

if ! command -v vulkaninfo >/dev/null 2>&1; then
  echo "ERROR: vulkaninfo is not installed" >&2
  exit 3
fi

export VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json
probe_output="$(vulkaninfo --summary 2>&1)" || {
  printf '%s\n' "${probe_output}" >&2
  echo "ERROR: Vulkan loader could not create an instance" >&2
  exit 4
}
printf '%s\n' "${probe_output}"

if ! grep -Eiq 'NVIDIA|RTX|GeForce|Quadro|Tesla' <<<"${probe_output}"; then
  echo "ERROR: Vulkan did not enumerate an NVIDIA GPU" >&2
  exit 5
fi

if grep -Eiq 'llvmpipe' <<<"${probe_output}"; then
  echo "ERROR: software Vulkan device llvmpipe is present" >&2
  exit 6
fi

echo "NVIDIA Vulkan preflight passed"
