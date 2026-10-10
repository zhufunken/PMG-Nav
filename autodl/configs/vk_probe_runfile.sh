#!/usr/bin/env bash
set -euo pipefail

driver_root=/root/autodl-tmp/nvidia-runfile-595
export LD_LIBRARY_PATH="$driver_root:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="$driver_root/nvidia_icd.json"

exec vulkaninfo --summary
