#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp
runfile="$root/NVIDIA-Linux-x86_64-595.71.05.run"
target="$root/nvidia-runfile-595"

wget -c \
  https://download.nvidia.com/XFree86/Linux-x86_64/595.71.05/NVIDIA-Linux-x86_64-595.71.05.run \
  -O "$runfile"
chmod +x "$runfile"
"$runfile" --extract-only --target "$target"

find "$target" -maxdepth 3 -type f \
  \( -name '*vulkan*' -o -name '*icd*.json' -o -name 'libEGL_nvidia*' \) \
  -print
