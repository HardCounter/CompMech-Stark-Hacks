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

## 4) Helpful next steps for your gripper project

- Add a data capture utility to save frames + labels from the Pi camera.
- Replace classifier with segmentation/detection model for grasp points.
- Export trained PyTorch models to ONNX for later TensorRT/edge optimization.
