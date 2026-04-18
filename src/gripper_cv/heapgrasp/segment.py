"""
Silhouette extraction for HEAPGrasp.

Three methods are provided:

  "background"  — Fast background subtraction. Requires a background frame
                  captured without the object. Good for a fixed camera + turntable.

  "deeplab"     — DeepLabv3+ (ResNet-50) semantic segmentation. Extracts
                  any non-background pixels. Heavier, but works without a
                  dedicated background frame and is more robust to lighting.
                  Uses torchvision.

  "finetuned"   — Same DeepLabv3+ backbone fine-tuned for 4-class transparent-
                  object segmentation. Requires a .pt checkpoint trained with
                  gripper-cv-train-seg.

  "hailo"       — Runs a compiled .hef model on the Hailo-8L NPU (AI Hat+).
                  Requires hef_path= and hailo_platform installed.
                  Export + compile flow: export_to_onnx() → hailo optimize → hailo compile
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .capture import CaptureSession


# ---------------------------------------------------------------------------
# Background subtraction
# ---------------------------------------------------------------------------

def _bg_subtract(frame_rgb: np.ndarray, background_rgb: np.ndarray, threshold: int) -> np.ndarray:
    diff = cv2.absdiff(frame_rgb, background_rgb).mean(axis=2)
    mask = (diff > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask > 0


# ---------------------------------------------------------------------------
# DeepLabv3+ (torchvision)
# ---------------------------------------------------------------------------

def _load_deeplab(device: str):
    import torch
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights).to(device).eval()
    return model, weights.transforms()


def _deeplab_mask(frame_rgb: np.ndarray, model, transform, device: str) -> np.ndarray:
    import torch
    from PIL import Image
    pil = Image.fromarray(frame_rgb)
    tensor = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)["out"][0]
    pred = out.argmax(0).cpu().numpy()
    return pred != 0   # class 0 = background in PASCAL VOC


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_finetuned_model(checkpoint_path: str, device: str):
    """
    Load the 4-class fine-tuned DeepLabv3+ saved by gripper-cv-train-seg.

    Returns (model, transform) with the same contract as _load_deeplab().
    The checkpoint can be a raw state-dict .pt or a dict with
    "model_state_dict" key (both formats are accepted).
    """
    import torch
    import torch.nn as nn
    from torchvision.models.segmentation import (
        DeepLabV3_ResNet50_Weights,
        deeplabv3_resnet50,
    )

    model = deeplabv3_resnet50(weights=None)
    in_ch = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_ch, 4, kernel_size=1)
    if model.aux_classifier is not None:
        in_ch_aux = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(in_ch_aux, 4, kernel_size=1)

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)

    transform = DeepLabV3_ResNet50_Weights.DEFAULT.transforms()
    return model.to(device).eval(), transform


def _finetuned_mask(
    frame_rgb: np.ndarray, model, transform, device: str
) -> np.ndarray:
    """Run the fine-tuned 4-class model; foreground = any predicted class != 0."""
    import torch
    from PIL import Image
    pil = Image.fromarray(frame_rgb)
    tensor = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)["out"][0]
    pred = out.argmax(0).cpu().numpy()
    return pred != 0


# ---------------------------------------------------------------------------
# Hailo-8L NPU (AI Hat+)
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _hailo_seg_mask(
    frame_rgb: np.ndarray,
    runner,
    img_size: tuple = (512, 512),
) -> np.ndarray:
    """
    Run segmentation on the Hailo-8L NPU.

    Args:
        frame_rgb: (H, W, 3) uint8 RGB input frame
        runner:    HailoRunner instance
        img_size:  (H, W) the model was compiled for

    Returns:
        bool mask — True where predicted class != 0 (foreground)
    """
    from PIL import Image
    H, W = img_size
    pil = Image.fromarray(frame_rgb).resize((W, H), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD         # (H, W, 3)
    chw = arr.transpose(2, 0, 1).astype(np.float32)      # (3, H, W) NCHW for HailoRunner

    output = runner.run_single(chw)   # HailoRunner returns NCHW: (1, num_classes, H, W)

    if output.ndim == 4:
        pred = output[0].argmax(0)    # (H_out, W_out)
    else:
        pred = output.argmax(-1)

    orig_h, orig_w = frame_rgb.shape[:2]
    if pred.shape != (orig_h, orig_w):
        pred = cv2.resize(
            pred.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
    return pred != 0


def extract_silhouettes(
    session: CaptureSession,
    method: str = "background",
    device: str = "cpu",
    bg_threshold: int = 30,
    checkpoint: str | None = None,
    hef_path: str | None = None,
    img_size: tuple = (512, 512),
) -> List[np.ndarray]:
    """
    Return a list of binary silhouette masks (H, W) bool, one per view.

    Args:
        session:       CaptureSession from capture_multiview()
        method:        "background", "deeplab", "finetuned", or "hailo"
        device:        torch device string (for deeplab / finetuned)
        bg_threshold:  pixel difference threshold (for background method)
        checkpoint:    path to .pt checkpoint (required for method="finetuned")
        hef_path:      path to compiled .hef model (required for method="hailo")
        img_size:      (H, W) the model was compiled for (hailo only)
    """
    masks: List[np.ndarray] = []

    if method == "background":
        if session.background is None:
            raise ValueError("Background frame missing. Re-run capture with object removed first.")
        for i, frame in enumerate(session.frames):
            masks.append(_bg_subtract(frame, session.background, bg_threshold))
            print(f"  Segmented view {i+1}/{len(session.frames)} (background subtraction)")

    elif method == "deeplab":
        print("Loading DeepLabv3+ (ResNet-50)…")
        model, transform = _load_deeplab(device)
        for i, frame in enumerate(session.frames):
            masks.append(_deeplab_mask(frame, model, transform, device))
            print(f"  Segmented view {i+1}/{len(session.frames)} (DeepLabv3+)")

    elif method == "finetuned":
        if checkpoint is None:
            raise ValueError(
                "method='finetuned' requires checkpoint= path to a .pt file. "
                "Train one with: gripper-cv-train-seg --output-dir outputs/seg"
            )
        print(f"Loading fine-tuned DeepLabv3+ from {checkpoint}…")
        model, transform = load_finetuned_model(checkpoint, device)
        for i, frame in enumerate(session.frames):
            masks.append(_finetuned_mask(frame, model, transform, device))
            print(f"  Segmented view {i+1}/{len(session.frames)} (finetuned DeepLabv3+)")

    elif method == "hailo":
        if hef_path is None:
            raise ValueError(
                "method='hailo' requires hef_path= to a compiled .hef file.\n"
                "Export + compile flow:\n"
                "  1. gripper-cv-export-onnx --checkpoint outputs/seg/best_model.pt --output model.onnx\n"
                "  2. hailo optimize model.onnx --hw-arch hailo8l --calib-path /calib\n"
                "  3. hailo compile model.har --hw-arch hailo8l --output-dir ."
            )
        from gripper_cv.hailo import HailoRunner
        print(f"Loading Hailo-8L NPU model from {hef_path}…")
        with HailoRunner(hef_path) as runner:
            for i, frame in enumerate(session.frames):
                masks.append(_hailo_seg_mask(frame, runner, img_size))
                print(f"  Segmented view {i+1}/{len(session.frames)} (Hailo-8L NPU)")

    else:
        raise ValueError(
            f"Unknown segmentation method '{method}'. "
            "Choose 'background', 'deeplab', 'finetuned', or 'hailo'."
        )

    return masks
