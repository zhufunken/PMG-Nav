#!/usr/bin/env bash
set -euo pipefail

GOAT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOAT_WORK_ROOT="${GOAT_WORK_ROOT:-/root/autodl-tmp/ccfb-work}"
GOAT_ENV_NAME="${GOAT_ENV_NAME:-ccfb-habitat023}"
GOAT_REPO_DIR="${GOAT_REPO_DIR:-${GOAT_WORK_ROOT}/goat-bench}"
EAI_VC_DIR="${EAI_VC_DIR:-${GOAT_WORK_ROOT}/eai-vc}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required" >&2
  exit 4
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -Fxq "$GOAT_ENV_NAME"; then
  echo "Shared Habitat environment is absent; installing OVON/common environment first."
  OVON_ENV_NAME="$GOAT_ENV_NAME" bash "$GOAT_SCRIPT_DIR/01_setup_ovon.sh"
fi

conda activate "$GOAT_ENV_NAME"
mkdir -p "$GOAT_WORK_ROOT"

if [[ ! -d "$GOAT_REPO_DIR/.git" ]]; then
  git clone https://github.com/Ram81/goat-bench.git "$GOAT_REPO_DIR"
fi

if [[ ! -d "$EAI_VC_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/facebookresearch/eai-vc.git "$EAI_VC_DIR"
fi

python -m pip install -e "$EAI_VC_DIR/vc_models"
python -m pip install -e "$GOAT_REPO_DIR"
python -m pip install -r "$GOAT_REPO_DIR/requirements.txt"
python -m pip install 'git+https://github.com/openai/CLIP.git'
python -m pip install \
  ftfy regex tqdm GPUtil trimesh seaborn timm scikit-learn einops transformers

python - <<'PY'
import torch
import habitat_sim
import habitat
import goat_bench
from vc_models.models.vit import model_utils

assert torch.cuda.is_available(), "CUDA is unavailable"
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("goat_imports_ok", model_utils.__name__)
PY

cat <<EOF
GOAT setup completed.
Environment: $GOAT_ENV_NAME
GOAT repo:   $GOAT_REPO_DIR
Next:        GOAT_DATA_ROOT=/root/autodl-tmp/ccfb-data bash scripts/04_smoke_goat.sh
EOF
