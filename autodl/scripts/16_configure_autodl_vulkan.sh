#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: run as root so the Vulkan ICD can be written under /etc" >&2
  exit 2
fi

export DEBIAN_FRONTEND=noninteractive
egl_library="$(ldconfig -p | awk '/libEGL_nvidia\.so\.0/{print $NF; exit}')"

if [[ -z "${egl_library}" || ! -e "${egl_library}" ]] || \
   ! command -v vulkaninfo >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends \
    libegl1 \
    libsm6 \
    libxt6 \
    libvulkan1 \
    vulkan-tools
  egl_library="$(ldconfig -p | awk '/libEGL_nvidia\.so\.0/{print $NF; exit}')"
fi

if [[ -z "${egl_library}" || ! -e "${egl_library}" ]]; then
  echo "ERROR: libEGL_nvidia.so.0 is not available for the active NVIDIA driver" >&2
  exit 3
fi

install -d /etc/vulkan/icd.d
cat > /etc/vulkan/icd.d/my_nvidia_icd.json <<EOF
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "${egl_library}",
    "api_version": "1.3.277"
  }
}
EOF

echo "Configured AutoDL headless NVIDIA Vulkan ICD: ${egl_library}"
echo "Source: https://www.autodl.com/docs/vulkan/"
