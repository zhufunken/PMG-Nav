#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/root/autodl-tmp/nvidia-vulkan-desktop/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="/root/nvidia2.json"

exec vulkaninfo --summary
