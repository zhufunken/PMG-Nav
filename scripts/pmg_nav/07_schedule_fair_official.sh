#!/usr/bin/env bash
set -euo pipefail

# Independent, reproducible entry point for the fair Stage 2 matrix. Keep
# this prefix separate from historical v3/tuned outputs so retries cannot
# silently change an already reported result.
export STAGE2_RUN_PREFIX="${STAGE2_RUN_PREFIX:-stage2_official_v5_fair}"
export STAGE2_PROGRESS_NAME="${STAGE2_PROGRESS_NAME:-v5_fair}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/07_schedule_tuned_official.sh" "$@"
