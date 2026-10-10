#!/usr/bin/env bash
set -euo pipefail

OVON_WORK_ROOT="${OVON_WORK_ROOT:-/root/autodl-tmp/ccfb-work}"
OVON_ENV_NAME="${OVON_ENV_NAME:-ccfb-habitat023}"
OVON_REPO_DIR="${OVON_REPO_DIR:-${OVON_WORK_ROOT}/ovon}"
OVON_DATA_ROOT="${OVON_DATA_ROOT:-/root/autodl-tmp/ccfb-data}"
OVON_SMOKE_STEPS="${OVON_SMOKE_STEPS:-3200}"
OVON_SMOKE_ENVS="${OVON_SMOKE_ENVS:-1}"
OVON_SMOKE_DIR="${OVON_SMOKE_DIR:-${OVON_WORK_ROOT}/smoke-output}"

eval "$(conda shell.bash hook)"
conda activate "$OVON_ENV_NAME"

python - <<'PY'
import torch
import habitat_sim
import habitat
import ovon

assert torch.cuda.is_available(), "CUDA is unavailable"
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("dependency_smoke_ok")
PY

required_data=(
  "$OVON_DATA_ROOT/scene_datasets/hm3d"
  "$OVON_DATA_ROOT/datasets/ovon/hm3d"
  "$OVON_DATA_ROOT/text_embeddings/siglip.pkl"
)

for required_item in "${required_data[@]}"; do
  if [[ ! -e "$required_item" ]]; then
    echo "ERROR: required data is missing: $required_item" >&2
    echo "Dependency smoke passed, but dataset/training smoke has not run." >&2
    exit 10
  fi
done

mkdir -p "$OVON_REPO_DIR/data" "$OVON_SMOKE_DIR/checkpoints" "$OVON_SMOKE_DIR/tb"

link_data_dir() {
  local source_dir="$1"
  local target_path="$2"
  if [[ -L "$target_path" ]]; then
    ln -sfn "$source_dir" "$target_path"
  elif [[ -e "$target_path" ]]; then
    echo "ERROR: refusing to replace existing non-symlink path: $target_path" >&2
    exit 12
  else
    ln -s "$source_dir" "$target_path"
  fi
}

link_data_dir "$OVON_DATA_ROOT/scene_datasets" "$OVON_REPO_DIR/data/scene_datasets"
link_data_dir "$OVON_DATA_ROOT/datasets" "$OVON_REPO_DIR/data/datasets"
link_data_dir "$OVON_DATA_ROOT/text_embeddings" "$OVON_REPO_DIR/data/text_embeddings"

cd "$OVON_REPO_DIR"

python -m ovon.run \
  --run-type train \
  --exp-config config/experiments/rnn_dagger.yaml \
  habitat_baselines.num_environments="$OVON_SMOKE_ENVS" \
  habitat_baselines.total_num_steps="$OVON_SMOKE_STEPS" \
  habitat_baselines.num_checkpoints=1 \
  habitat_baselines.checkpoint_folder="$OVON_SMOKE_DIR/checkpoints" \
  habitat_baselines.tensorboard_dir="$OVON_SMOKE_DIR/tb"

if ! find "$OVON_SMOKE_DIR/checkpoints" -type f -print -quit | grep -q .; then
  echo "ERROR: training command exited but no checkpoint was created" >&2
  exit 11
fi

echo "OVON dataset/training smoke passed: $OVON_SMOKE_STEPS steps, $OVON_SMOKE_ENVS environment(s)."
