#!/usr/bin/env bash
set -euo pipefail

ticvla_work_root="${TICVLA_WORK_ROOT:-/root/autodl-tmp/ccfb-ticvla}"
ticvla_repo_dir="${TICVLA_REPO_DIR:-${ticvla_work_root}/src/TIC-VLA}"
isaaclab_root="${ISAACLAB_ROOT:-${ticvla_work_root}/IsaacLab}"

for required_path in \
  "${ticvla_repo_dir}/rl/fine_tuning/train_rl.py" \
  "${isaaclab_root}/isaaclab.sh" \
  "${TICVLA_BASE_MODEL_PATH:-}" \
  "${TICVLA_CHECKPOINT_PATH:-}"; do
  if [[ -z "${required_path}" || ! -e "${required_path}" ]]; then
    echo "ERROR: required path is missing: ${required_path:-unset}" >&2
    exit 2
  fi
done

export TICVLA_ROOT="${ticvla_repo_dir}"
export TICVLA_RL_LOG_DIR="${TICVLA_RL_LOG_DIR:-${ticvla_work_root}/outputs/rl_logs}"
mkdir -p "${TICVLA_RL_LOG_DIR}"

cd "${ticvla_repo_dir}"
"${isaaclab_root}/isaaclab.sh" -p "${ticvla_repo_dir}/rl/fine_tuning/train_rl.py" \
  --task Isaac-Navigation-TICVLA-COCO \
  --enable_cameras \
  --model_path "${TICVLA_BASE_MODEL_PATH}" \
  --checkpoint "${TICVLA_CHECKPOINT_PATH}" \
  --max_iterations "${TICVLA_RL_ITERATIONS:-20}" \
  --rollout_steps "${TICVLA_RL_ROLLOUT_STEPS:-128}" \
  --save_interval "${TICVLA_RL_SAVE_INTERVAL:-10}"

