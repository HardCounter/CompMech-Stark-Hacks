# Gripper CV Starter (Raspberry Pi 5)

This repository now includes a clean Python starter project for computer vision on a Raspberry Pi 5, designed for your robotic gripper workflow.

It is **PyTorch-first** for ML experiments and uses `picamera2` for camera capture.

## What is included

- `src/gripper_cv/camera.py`  
  Shared `picamera2` camera wrapper with RGB/BGR frame APIs.

- `src/gripper_cv/app_preview.py`  
  Live OpenCV preview with FPS overlay.

- `src/gripper_cv/app_mjpeg.py`  
  MJPEG HTTP stream server (equivalent to your current debug stream flow).

- `src/gripper_cv/app_classifier.py`  
  Real-time PyTorch image classification demo (MobileNetV3).

- `scripts/setup_venv.sh`  
  Pi-friendly venv bootstrap script.

## 1) Raspberry Pi prerequisites

Install system camera libs once:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv libatlas-base-dev
```

## 2) Create and configure venv

From repo root:

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
```

Notes:

- The setup script uses `--system-site-packages` by default so your venv can access apt-installed `picamera2` and `cv2`.
- To disable that behavior:
  ```bash
  USE_SYSTEM_SITE_PACKAGES=0 ./scripts/setup_venv.sh
  ```

## 3) Run starter apps

Camera preview:

```bash
gripper-cv-preview --width 640 --height 480 --fps 30
```

MJPEG stream:

```bash
gripper-cv-mjpeg --host 0.0.0.0 --port 8000 --fps 20
```

Then open:

`http://<PI_IP>:8000`

PyTorch classification demo:

```bash
gripper-cv-classify --fps 20 --device cpu
```

## 4) HEAPGrasp: shape-from-silhouette + grasp planning

The `gripper_cv.heapgrasp` package implements the full HEAPGrasp pipeline:
capture → auto-calibrate (ArUco) → segment → voxel carve (visual hull) →
PLY export → parallel-jaw grasp ranking → `grasps.json`.

### CLI

Run the end-to-end pipeline from the Pi:

```bash
gripper-cv-heapgrasp \
    --out outputs/heapgrasp \
    --n-views 12 \
    --volume 64 \
    --segment background \
    --top-k-grasps 5
```

The CLI writes `reconstruction.ply`, `voxels.npy`, `masks/*.png`, and
`grasps.json` into the chosen output directory. See
`gripper-cv-heapgrasp --help` for all flags, including segmentation backends
(`background`, `deeplab`, `finetuned`, `hailo`) and the optional learned
grasp scorer (`--grasp-onnx`, `--grasp-hef`).

### Dashboard

A Streamlit dashboard ships alongside the CLI for interactive exploration:

```bash
pip install -e .[dashboard]
streamlit run dashboard/app.py
```

Features: live capture wizard, segmentation playground, 3D object scan with
the Next-Best-View planner, Hailo-8L status page, and a theoretical grasp
plan that shares the same grasp module as the CLI.

### Grasp module

Structured grasp candidates live in
`gripper_cv.heapgrasp.grasp` (geometric, PCA-based) and
`gripper_cv.heapgrasp.grasp_learned` (GQ-CNN-style scorer over ONNX
Runtime or Hailo). Both produce `GraspCandidate` dataclasses with
`position`, `approach`, `jaw_axis`, `width`, and `score`.

### Dex-Net 2.0 / GQ-CNN 2.0 integration

The learned scorer can consume stock Dex-Net 2.0 GQ-CNN 2.0 weights via ONNX.
The patch renderer supports both the internal normalised `[0, 1]` encoding
and the Dex-Net metric encoding (raw depth in metres, centred on the grasp
mid-plane, with median fill for empty cells), and the ONNX backend
transparently handles the Dex-Net dual-input signature
(`image` + `pose` gripper-depth scalar) with automatic NCHW/NHWC layout
detection.

Produce an ONNX model with the helper script:

```bash
# Preferred: tf2onnx from Berkeley's unpacked checkpoint (TF1 + tf2onnx needed)
python scripts/export_gqcnn_onnx.py \
    --strategy tf2onnx \
    --checkpoint path/to/GQ-Image-Wise \
    --output models/gqcnn_2.0.onnx

# Fallback: PyTorch re-implementation with random init (needs only torch)
python scripts/export_gqcnn_onnx.py \
    --strategy pytorch \
    --output models/gqcnn_2.0.onnx
```

Then run the pipeline with the preset flag. `--grasp-preset auto` will read
the ONNX metadata emitted by the export script and pick the correct
conventions automatically:

```bash
gripper-cv-heapgrasp \
    --grasp-onnx models/gqcnn_2.0.onnx \
    --grasp-preset auto \
    --top-k-grasps 5
```

**Domain gap caveat.** The scorer feeds the network patches rendered from
the **visual hull** (silhouette carving), not from a real depth camera.
Stock Dex-Net 2.0 weights were trained on synthetic depth-image patches, so
the rankings will be noisy on our visual-hull patches — this is expected for
the hackathon demo. The structural pipeline (rendering, dual-input feeds,
ranking, `grasps.json` export) is exactly what a fine-tuned model will
consume. Plan B is to fine-tune on patches rendered from our own pipeline
once compute is online (see `scripts/train_gqcnn.py`).

**NPU acceleration (optional, Phase C).** `scripts/compile_gqcnn_hef.sh`
documents the Hailo Dataflow Compiler invocation that produces a `.hef`
from the exported ONNX. Run it on a Linux workstation with the Hailo DFC
installed, copy the `.hef` to the Pi, and load it via `--grasp-hef`. The
script flags known risks (dual-input support in older DFC versions); the
default CPU ONNX Runtime path is fine for the demo if the compile fails.

### Optional extras

Declared in `pyproject.toml`:

- `[dashboard]` — Streamlit, Plotly, matplotlib.
- `[ml]` — PyTorch + torchvision + ONNX Runtime (for the fine-tuned
  segmenter and learned grasp scorer on CPU).
- `[hailo]` — `hailo-platform` for NPU inference.

### Tests

```bash
pytest -q
```

Covers the SfS core, PLY/npy/JSON export, grasp module, patch renderer,
NBV planner, and Hailo helpers.

## 5) Helpful next steps for your gripper project

- Add a data capture utility to save frames + labels from the Pi camera.
- Replace classifier with segmentation/detection model for grasp points.
- Export trained PyTorch models to ONNX for later TensorRT/edge optimization.
