from __future__ import annotations

import argparse

import numpy as np
from gripper_cv.utils import FpsMeter


IMAGENET_LABELS = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead", "electric ray", "stingray",
    "cock", "hen", "ostrich",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time PyTorch classification demo on Pi camera.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def build_model(device, models):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.eval()
    model.to(device)
    return model


def build_transform(transforms):
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict(model, transform, frame_rgb: np.ndarray, device, torch, image_cls):
    image = image_cls.fromarray(frame_rgb)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        score, index = torch.max(probs, dim=1)
    return int(index.item()), float(score.item())


def class_name(index: int) -> str:
    if 0 <= index < len(IMAGENET_LABELS):
        return IMAGENET_LABELS[index]
    return f"class_{index}"


def main() -> None:
    args = parse_args()

    from gripper_cv.camera import CameraConfig, PiCameraStream

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV (cv2) is required for classifier overlay. "
            "Install on Raspberry Pi OS with: sudo apt install -y python3-opencv"
        ) from exc
    try:
        import torch
        from PIL import Image
        from torchvision import models, transforms
    except ImportError as exc:
        raise ImportError(
            "PyTorch stack is required for classification. "
            "Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_model(device, models)
    transform = build_transform(transforms)
    fps_meter = FpsMeter()
    cfg = CameraConfig(width=args.width, height=args.height, fps=args.fps)

    with PiCameraStream(cfg) as cam:
        while True:
            frame_bgr = cam.read_bgr()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            idx, conf = predict(model, transform, frame_rgb, device, torch, Image)
            fps = fps_meter.tick()
            label = class_name(idx)
            text = f"{label} ({conf:.2f}) | FPS {fps:.1f}"
            cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("gripper-cv-classifier", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
