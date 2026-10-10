#!/usr/bin/env bash
set -euo pipefail

# Bounded Stage-2 geometry gate.  This is intentionally a single-episode,
# short smoke runner; it must not be used as a full benchmark launcher.
root="${TICVLA_WORK_ROOT:-/root/autodl-tmp/ccfb-ticvla}"
project_main="${PROJECT_MAIN:-${root}/project_gate_latch/main}"
run_id="${TICVLA_DYNANAV_RUN_ID:?set a unique run id}"
config_path="${1:?usage: run_radius_gate.sh CONFIG [SMOKE_FRAMES]}"
smoke_frames="${2:-300}"

cd "${project_main}/autodl"
export TICVLA_WORK_ROOT="${root}"
export PROJECT_MAIN="${project_main}"
export PMG_ROOT="${project_main}"
export TICVLA_REPO_DIR="${TICVLA_REPO_DIR:-${root}/src/TIC-VLA}"
export ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-${root}/isaacsim-6.0.1}"
export ISAAC_SIM_PYTHON="${ISAAC_SIM_PYTHON:-${ISAAC_SIM_ROOT}/python.sh}"
export TICVLA_BASE_MODEL_PATH="${TICVLA_BASE_MODEL_PATH:-${root}/models/InternVL3-1B}"
export TICVLA_CHECKPOINT_PATH="${TICVLA_CHECKPOINT_PATH:-${root}/data/TIC-VLA/TIC-VLA-model.ckpt}"
export TICVLA_DATA_ROOT="${TICVLA_DATA_ROOT:-${root}/data/TIC-VLA}"
export TICVLA_ISAAC6_COMPAT=1
export TICVLA_DYNANAV_RUN_ID="${run_id}"
export PMG_METHOD="${PMG_METHOD:-M1}"
export STAGE2_M1_VARIANT="${STAGE2_M1_VARIANT:-simple_oracle}"
export PMG_GRAPH_MODE="${PMG_GRAPH_MODE:-compact}"
export PMG_MAX_GRAPH_NODES="${PMG_MAX_GRAPH_NODES:-8}"
export PMG_MAX_EDGES_PER_ANCHOR="${PMG_MAX_EDGES_PER_ANCHOR:-4}"
export PMG_CONTROL_POLICY="${PMG_CONTROL_POLICY:-global}"
export PMG_ROUTE_PLANNER="${PMG_ROUTE_PLANNER:-1}"
export PMG_ROUTE_ROBOT_RADIUS_M="${PMG_ROUTE_ROBOT_RADIUS_M:-0.55}"
export PMG_ROUTE_WAYPOINT_STOP_RADIUS="${PMG_ROUTE_WAYPOINT_STOP_RADIUS:-0.20}"
# Optional anisotropic footprint additions.  They are zero by default and
# only enabled explicitly for a geometry gate (e.g. x=0.50 for Carter's
# offset root); this keeps the ordinary M1/M0 path unchanged.
export PMG_ROUTE_AXIS_CLEARANCE_X_M="${PMG_ROUTE_AXIS_CLEARANCE_X_M:-0.0}"
export PMG_ROUTE_AXIS_CLEARANCE_Y_M="${PMG_ROUTE_AXIS_CLEARANCE_Y_M:-0.0}"
export PMG_ENABLE_GOAL_ADAPTER=0
export PMG_ENABLE_VECTOR_ADAPTER=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/my_nvidia_icd.json}"

# The compatibility runner intentionally keeps strict replay opt-in behind
# the explicit parity aliases.  Propagate the matrix-level switch here so a
# bounded radius-gate invocation cannot silently run stochastic/early-success
# behavior while being labeled as strict parity.
if [[ "${STAGE2_STRICT_PARITY:-0}" == "1" || "${PMG_STRICT_PARITY:-0}" == "1" ]]; then
  export PMG_PARITY_FIXED_HORIZON=1
  export STAGE2_PARITY_FIXED_HORIZON=1
  export PMG_PARITY_DETERMINISTIC=1
  export STAGE2_PARITY_DETERMINISTIC=1
  export STAGE2_DISABLE_STUCK_RECOVERY=1
fi

bash scripts/08_run_dynanav_baseline.sh "${config_path}" --smoke-frames "${smoke_frames}"
