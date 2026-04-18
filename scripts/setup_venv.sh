#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_SYSTEM_SITE_PACKAGES="${USE_SYSTEM_SITE_PACKAGES:-1}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"

echo "==> Python: $($PYTHON_BIN --version)"
echo "==> Creating virtualenv at: ${VENV_DIR}"

if [[ "${USE_SYSTEM_SITE_PACKAGES}" == "1" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "${VENV_DIR}"
else
  "$PYTHON_BIN" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .

if [[ "${INSTALL_TORCH}" == "1" ]]; then
  echo "==> Installing PyTorch stack"
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo
echo "Setup complete."
echo "Activate: source ${VENV_DIR}/bin/activate"
echo "Preview:  gripper-cv-preview --width 640 --height 480 --fps 30"
echo "Stream:   gripper-cv-mjpeg --port 8000"
echo "Classify: gripper-cv-classify --fps 20 --device cpu"
