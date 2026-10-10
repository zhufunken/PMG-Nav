#!/usr/bin/env bash
set -euo pipefail

echo "=== system ==="
date -Is
uname -a
cat /etc/os-release 2>/dev/null || true

echo "=== gpu ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,compute_cap --format=csv,noheader 2>/dev/null || \
    nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader
else
  echo "ERROR: nvidia-smi is unavailable" >&2
  exit 2
fi

echo "=== cpu-memory-disk ==="
lscpu
free -h
df -hT

echo "=== toolchain ==="
for tool_name in python python3 conda mamba git gcc g++ cmake nvcc docker; do
  if command -v "$tool_name" >/dev/null 2>&1; then
    printf '%-10s %s\n' "$tool_name" "$(command -v "$tool_name")"
    "$tool_name" --version 2>&1 | head -n 2 || true
  else
    printf '%-10s missing\n' "$tool_name"
  fi
done

echo "=== network-readiness ==="
for endpoint_url in \
  https://github.com \
  https://huggingface.co \
  https://download.pytorch.org \
  https://api.anaconda.org; do
  status_code="$(curl -LIsS --max-time 15 -o /dev/null -w '%{http_code}' "$endpoint_url" 2>/dev/null || true)"
  printf '%-40s %s\n' "$endpoint_url" "${status_code:-unreachable}"
done

echo "=== probe-complete ==="
