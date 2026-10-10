#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/root/autodl-tmp/nvidia-vulkan/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="/root/nvidia.json"

exec vulkaninfo --summary
