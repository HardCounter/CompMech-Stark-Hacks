#!/usr/bin/env bash
# Phase C (optional): attempt to compile the GQ-CNN 2.0 ONNX into a Hailo .hef
# for NPU-accelerated grasp scoring on the Pi 5 + AI HAT+.
#
# This script is a documented attempt — it is NOT run during the hackathon
# demo because the Hailo Dataflow Compiler is Linux-only and distributed
# outside PyPI (requires a Hailo developer account + .deb packages).
#
# Run this on a Linux workstation (Ubuntu 22.04 recommended) that has the
# Hailo Dataflow Compiler 3.x installed, then copy the resulting .hef to the
# Raspberry Pi and load it with ``--grasp-hef models/gqcnn_2.0.hef``.
#
# Known risks
# -----------
# 1. GQ-CNN 2.0 has **two inputs** (image + pose). The Hailo compiler prefers
#    single-input graphs. If this compile fails, the recommended workaround is
#    to bake the pose scalar in as a constant at export time (set a fixed
#    camera height) and re-export a single-input variant, then retry.
# 2. The model has two input tensors with different ranks; some Hailo versions
#    require identical ranks — we may need to tile the pose into a (1,1,1,1)
#    tensor to satisfy the compiler.
# 3. ONNX opset mismatch: Hailo DFC 3.x fully supports opsets 10–13. The
#    exporter uses opset 13, which should be compatible.
#
# Outcome is recorded in README.md under the HEAPGrasp section. Fall back to
# CPU ONNX Runtime on the Pi 5 if the compile fails — the model is small
# enough (~18M params) that CPU inference is fine for the demo.

set -euo pipefail

ONNX=${1:-models/gqcnn_2.0.onnx}
OUT_DIR=${2:-models}
ARCH=${HAILO_ARCH:-hailo8l}   # Pi 5 + AI HAT+ ships the Hailo-8L

if ! command -v hailo >/dev/null 2>&1; then
    echo "error: Hailo DFC (hailo CLI) not found on PATH." >&2
    echo "Install from https://hailo.ai/developer-zone/software-downloads/ and" >&2
    echo "activate the dfc venv before running this script." >&2
    exit 2
fi

BASENAME=$(basename "${ONNX}" .onnx)
HAR_OPTIMIZED="${OUT_DIR}/${BASENAME}_optimized.har"
HEF_OUT="${OUT_DIR}/${BASENAME}.hef"

echo "==> Parsing ${ONNX} to Hailo HAR"
hailo parser onnx "${ONNX}" \
    --hw-arch "${ARCH}" \
    --har-path "${OUT_DIR}/${BASENAME}.har"

echo "==> Optimising (quantisation)"
# Provide a small calibration set if you have one; random works for a demo.
hailo optimize "${OUT_DIR}/${BASENAME}.har" \
    --output-har-path "${HAR_OPTIMIZED}" \
    --hw-arch "${ARCH}"

echo "==> Compiling to ${HEF_OUT}"
hailo compiler "${HAR_OPTIMIZED}" \
    --output-dir "${OUT_DIR}" \
    --hw-arch "${ARCH}"

echo "==> Done. Copy ${HEF_OUT} to the Pi and run:"
echo "     gripper-cv-heapgrasp --grasp-hef ${HEF_OUT} --grasp-preset gqcnn2"
