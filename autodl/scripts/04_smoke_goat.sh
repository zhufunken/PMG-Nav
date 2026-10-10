#!/usr/bin/env bash
set -euo pipefail

GOAT_WORK_ROOT="${GOAT_WORK_ROOT:-/root/autodl-tmp/ccfb-work}"
GOAT_ENV_NAME="${GOAT_ENV_NAME:-ccfb-habitat023}"
GOAT_REPO_DIR="${GOAT_REPO_DIR:-${GOAT_WORK_ROOT}/goat-bench}"
GOAT_DATA_ROOT="${GOAT_DATA_ROOT:-/root/autodl-tmp/ccfb-data}"
GOAT_SMOKE_UPDATES="${GOAT_SMOKE_UPDATES:-1}"
GOAT_SMOKE_ENVS="${GOAT_SMOKE_ENVS:-1}"
GOAT_SMOKE_DIR="${GOAT_SMOKE_DIR:-${GOAT_WORK_ROOT}/goat-smoke-output}"

eval "$(conda shell.bash hook)"
conda activate "$GOAT_ENV_NAME"

python - <<'PY'
import torch
import habitat_sim
import habitat
import goat_bench

assert torch.cuda.is_available(), "CUDA is unavailable"
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("goat_dependency_smoke_ok")
PY

required_data=(
  "$GOAT_DATA_ROOT/scene_datasets/hm3d"
  "$GOAT_DATA_ROOT/datasets/goat_bench/v1/train/train.json.gz"
  "$GOAT_DATA_ROOT/goat-assets/goal_cache/ovon/category_name_clip_embeddings.pkl"
  "$GOAT_DATA_ROOT/goat-assets/goal_cache/iin/train_embeddings"
  "$GOAT_DATA_ROOT/goat-assets/goal_cache/language_nav/train_instruction_clip_embeddings.pkl"
)

for required_item in "${required_data[@]}"; do
  if [[ ! -e "$required_item" ]]; then
    echo "ERROR: required data is missing: $required_item" >&2
    echo "Dependency smoke passed, but GOAT dataset/training smoke has not run." >&2
    exit 20
  fi
done

mkdir -p "$GOAT_REPO_DIR/data" "$GOAT_SMOKE_DIR/checkpoints" "$GOAT_SMOKE_DIR/tb"

link_data_dir() {
  local source_dir="$1"
  local target_path="$2"
  if [[ -L "$target_path" ]]; then
    ln -sfn "$source_dir" "$target_path"
  elif [[ -e "$target_path" ]]; then
    echo "ERROR: refusing to replace existing non-symlink path: $target_path" >&2
    exit 21
  else
    ln -s "$source_dir" "$target_path"
  fi
}

link_data_dir "$GOAT_DATA_ROOT/scene_datasets" "$GOAT_REPO_DIR/data/scene_datasets"
link_data_dir "$GOAT_DATA_ROOT/datasets" "$GOAT_REPO_DIR/data/datasets"
link_data_dir "$GOAT_DATA_ROOT/goat-assets" "$GOAT_REPO_DIR/data/goat-assets"

cd "$GOAT_REPO_DIR"

python -u -m goat_bench.run \
  --run-type train \
  --exp-config config/experiments/ver_goat_monolithic.yaml \
  habitat_baselines.num_environments="$GOAT_SMOKE_ENVS" \
  habitat_baselines.num_updates="$GOAT_SMOKE_UPDATES" \
  habitat_baselines.num_checkpoints=1 \
  habitat_baselines.checkpoint_folder="$GOAT_SMOKE_DIR/checkpoints" \
  habitat_baselines.tensorboard_dir="$GOAT_SMOKE_DIR/tb"

if ! find "$GOAT_SMOKE_DIR/checkpoints" -type f -print -quit | grep -q .; then
  echo "ERROR: training command exited but no checkpoint was created" >&2
  exit 22
fi

echo "GOAT training smoke passed: $GOAT_SMOKE_UPDATES update(s), $GOAT_SMOKE_ENVS environment(s)."
