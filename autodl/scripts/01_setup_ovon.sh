#!/usr/bin/env bash
set -euo pipefail

OVON_WORK_ROOT="${OVON_WORK_ROOT:-/root/autodl-tmp/ccfb-work}"
OVON_ENV_NAME="${OVON_ENV_NAME:-ccfb-habitat023}"
OVON_REPO_DIR="${OVON_REPO_DIR:-${OVON_WORK_ROOT}/ovon}"
HABITAT_LAB_DIR="${HABITAT_LAB_DIR:-${OVON_WORK_ROOT}/habitat-lab}"
OVON_GPU_PROFILE="${OVON_GPU_PROFILE:-auto}"

mkdir -p "$OVON_WORK_ROOT"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable" >&2
  exit 2
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "Detected GPU: $GPU_NAME"

if [[ "$OVON_GPU_PROFILE" == "auto" ]]; then
  case "$GPU_NAME" in
    *5090*|*Blackwell*|*PRO\ 6000*|*6000D*) OVON_GPU_PROFILE="blackwell" ;;
    *4090*|*Ada*) OVON_GPU_PROFILE="ada" ;;
    *) OVON_GPU_PROFILE="legacy" ;;
  esac
fi

if [[ "$OVON_GPU_PROFILE" == "blackwell" ]]; then
  cat >&2 <<'EOF'
ERROR: Blackwell requires a CUDA 12.8+ PyTorch build, while the official OVON stack pins
PyTorch 1.12.1/CUDA 11.3 and Habitat 0.2.3. Do not spend rental time debugging this
combination. Use V100/A800/3090, or 4090 with OVON_GPU_PROFILE=ada.
EOF
  exit 3
fi

if command -v mamba >/dev/null 2>&1; then
  CONDA_RUNNER="mamba"
elif command -v conda >/dev/null 2>&1; then
  CONDA_RUNNER="conda"
else
  echo "ERROR: conda or mamba is required. Select an AutoDL Conda/PyTorch base image." >&2
  exit 4
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -Fxq "$OVON_ENV_NAME"; then
  if [[ "$OVON_GPU_PROFILE" == "legacy" ]]; then
    "$CONDA_RUNNER" create -n "$OVON_ENV_NAME" python=3.7 cmake=3.14.0 -y
    "$CONDA_RUNNER" install -n "$OVON_ENV_NAME" \
      habitat-sim=0.2.3 headless \
      pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=11.3 \
      -c pytorch -c nvidia -c conda-forge -c aihabitat -y
  elif [[ "$OVON_GPU_PROFILE" == "ada" ]]; then
    "$CONDA_RUNNER" create -n "$OVON_ENV_NAME" python=3.8 cmake=3.14.0 -y
    "$CONDA_RUNNER" install -n "$OVON_ENV_NAME" \
      pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.8 \
      -c pytorch -c nvidia -y
    "$CONDA_RUNNER" install -n "$OVON_ENV_NAME" habitat-sim=0.2.3 headless \
      -c conda-forge -c aihabitat -y
  else
    echo "ERROR: unknown OVON_GPU_PROFILE=$OVON_GPU_PROFILE" >&2
    exit 5
  fi
fi

conda activate "$OVON_ENV_NAME"
python -m pip install --upgrade 'pip<24.1' 'setuptools<70' wheel

if [[ ! -d "$OVON_REPO_DIR/.git" ]]; then
  git clone https://github.com/naokiyokoyama/ovon.git "$OVON_REPO_DIR"
fi

if [[ ! -d "$HABITAT_LAB_DIR/.git" ]]; then
  git clone --branch v0.2.3 --depth 1 https://github.com/facebookresearch/habitat-lab.git "$HABITAT_LAB_DIR"
fi

python -m pip install -e "$HABITAT_LAB_DIR/habitat-lab"
python -m pip install -e "$HABITAT_LAB_DIR/habitat-baselines"
python -m pip install -e "$OVON_REPO_DIR"
python -m pip install -e "$OVON_REPO_DIR/frontier_exploration"
python -m pip install \
  ftfy regex tqdm GPUtil trimesh seaborn timm scikit-learn einops transformers
python -m pip install 'git+https://github.com/openai/CLIP.git'

python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print("gpu", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("cuda_matmul_ok", float(y[0, 0]))

import habitat_sim
import habitat
import ovon
print("habitat_imports_ok")
PY

cat <<EOF
Setup completed.
Environment: $OVON_ENV_NAME
Profile:     $OVON_GPU_PROFILE
OVON repo:   $OVON_REPO_DIR
Next:        OVON_DATA_ROOT=/root/autodl-tmp/ccfb-data bash scripts/02_smoke_ovon.sh
EOF
