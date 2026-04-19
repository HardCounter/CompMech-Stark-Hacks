#!/usr/bin/env bash
# Bootstrap script for a freshly provisioned Azure GPU VM.
# Intended to run on the VM itself, as the admin user, once over SSH.
#
# Assumes the image is microsoft-dsvm:ubuntu-hpc:2204:latest, which ships
# with CUDA + NVIDIA drivers preinstalled.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/CompMech-Stark-Hacks/CompMech-Stark-Hacks.git}"
REPO_DIR="${REPO_DIR:-$HOME/CompMech-Stark-Hacks}"
PY="${PY:-python3.11}"

log() { printf '\033[1;34m[bootstrap]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[bootstrap]\033[0m %s\n' "$*" >&2; exit 1; }

log "system update"
sudo apt-get update -qq
sudo apt-get install -y -qq git curl ca-certificates build-essential

log "ensure python ${PY}"
if ! command -v "${PY}" >/dev/null 2>&1; then
    sudo apt-get install -y -qq software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
fi

log "verify NVIDIA driver (should be preinstalled on ubuntu-hpc image)"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found. Image is wrong, or drivers missing. Re-provision with microsoft-dsvm:ubuntu-hpc:2204:latest."
fi
nvidia-smi | head -n 20

log "clone repo"
if [ ! -d "${REPO_DIR}/.git" ]; then
    git clone "${REPO_URL}" "${REPO_DIR}"
else
    log "  already cloned; pulling latest"
    git -C "${REPO_DIR}" pull --ff-only
fi
cd "${REPO_DIR}"

log "create venv"
if [ ! -d .venv ]; then
    "${PY}" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip wheel

log "install project (editable) + CUDA torch"
# Torch CUDA wheels: pick the index that matches the preinstalled CUDA runtime.
# ubuntu-hpc:2204 currently ships CUDA 12.x drivers -> cu121 torch wheels.
pip install -q --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121
pip install -q -e ".[dev,dexnet]"

log "sanity check: torch sees the GPU"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print(f"  torch {torch.__version__}  cuda {torch.version.cuda}  gpu {torch.cuda.get_device_name(0)}")
print(f"  vram {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PY

log "ONNX export smoke test"
python scripts/export_gqcnn_onnx.py --strategy pytorch --output models/gqcnn_2.0.onnx

log "pytest (CPU-only, should still pass on GPU host)"
python -m pytest tests/ -q

log "DONE. Activate with:   cd ${REPO_DIR} && source .venv/bin/activate"
log "Ready for Phase B once --data-dir is wired in scripts/train_gqcnn.py."
