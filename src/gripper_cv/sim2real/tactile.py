"""
Slip detection for tactile sensors (GelSight-compatible).

SlipDetectorCNN — small CNN (3 conv + 2 FC) for binary slip classification.
  Input:  (H, W, 3) uint8 RGB tactile image
  Output: P(slipping) in [0, 1]

SlipDetector — inference wrapper; loads a checkpoint saved by train_slip_detector().

train_slip_detector — trains from a directory:
  data_dir/
    slipping/   *.jpg / *.png    label = 1.0
    stable/     *.jpg / *.png    label = 0.0

All torch imports are lazy (inside function/method bodies only).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# Model builder — lazy torch
# ---------------------------------------------------------------------------

def _build_cnn():
    """Build the SlipDetectorCNN as a torch nn.Sequential (torch imported lazily)."""
    import torch.nn as nn

    return nn.Sequential(
        # Block 1 — (3, H, W) → (16, H/4, W/4)
        nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Block 2 — (16, H/4, W/4) → (32, H/8, W/8)
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        # Block 3 — (32, H/8, W/8) → (64, 4, 4)
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((4, 4)),
        # Classifier
        nn.Flatten(),
        nn.Linear(64 * 16, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        # Raw logit — use BCEWithLogitsLoss during training; sigmoid at inference
    )


# ---------------------------------------------------------------------------
# SlipDetector — inference wrapper
# ---------------------------------------------------------------------------

class SlipDetector:
    """
    Inference wrapper for the slip-detection CNN.

    Runs on CPU/CUDA (PyTorch) by default.  Pass hef_path to route inference
    through the Hailo-8L NPU instead (requires hailo_platform installed and a
    .hef compiled from the ONNX export).

    Args:
        checkpoint_path:  path to .pt file saved by train_slip_detector()
                          (ignored when hef_path is provided)
        device:           "cpu" or "cuda" (PyTorch path only)
        threshold:        P(slipping) threshold; default 0.5
        hef_path:         optional path to a compiled .hef for Hailo-8L NPU
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        threshold: float = 0.5,
        hef_path: str | Path | None = None,
    ) -> None:
        self._threshold = threshold
        self._hailo_runner = None

        if hef_path is not None:
            from gripper_cv.hailo import HailoRunner
            self._hailo_runner = HailoRunner(hef_path)
        else:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required when hef_path is not provided.")
            import torch
            self._device = torch.device(device)
            self._model = _build_cnn().to(self._device)
            state = torch.load(
                Path(checkpoint_path), map_location=self._device, weights_only=True
            )
            self._model.load_state_dict(state)
            self._model.eval()

    def predict(self, gelsight_rgb: "np.ndarray") -> Tuple[bool, float]:
        """
        Classify a GelSight-style tactile image as slipping or stable.

        Args:
            gelsight_rgb: (H, W, 3) uint8 RGB array (from PiCamera or GelSight)

        Returns:
            (is_slipping: bool, probability: float)
            probability >= threshold → is_slipping=True
        """
        import numpy as np
        from PIL import Image

        pil = Image.fromarray(gelsight_rgb).resize((128, 128))
        arr = np.array(pil, dtype=np.float32) / 255.0
        mean = np.array(self._IMAGENET_MEAN, dtype=np.float32)
        std = np.array(self._IMAGENET_STD, dtype=np.float32)
        arr = (arr - mean) / std  # (128, 128, 3)

        if self._hailo_runner is not None:
            # Hailo path — HailoRunner expects NCHW, returns (1, 1) logit
            chw = arr.transpose(2, 0, 1).astype(np.float32)  # (3, 128, 128)
            output = self._hailo_runner.run_single(chw).ravel()
            prob = float(1.0 / (1.0 + np.exp(-output[0])))   # sigmoid
        else:
            import torch
            tensor = (
                torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device)
            )
            with torch.inference_mode():
                logit = self._model(tensor)
                prob = float(torch.sigmoid(logit).item())

        return prob >= self._threshold, prob

    def close(self) -> None:
        """Release the Hailo runner if one was opened."""
        if self._hailo_runner is not None:
            self._hailo_runner.close()

    def __enter__(self) -> "SlipDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# train_slip_detector
# ---------------------------------------------------------------------------

def train_slip_detector(
    data_dir: str | Path,
    output_path: str | Path,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_split: float = 0.2,
    device: str = "cpu",
    image_size: Tuple[int, int] = (128, 128),
) -> None:
    """
    Train SlipDetectorCNN from a directory of labelled tactile images.

    Expected layout:
        data_dir/
            slipping/   *.jpg / *.png   → label 1 (slipping)
            stable/     *.jpg / *.png   → label 0 (stable)

    Saves the best model (by validation accuracy) to output_path.

    All torch imports are inside this function body.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    from pathlib import Path as _Path
    from PIL import Image as _Image
    from torch.utils.data import DataLoader, Dataset, random_split

    data_dir = _Path(data_dir)
    output_path = _Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    slip_paths = sorted(p for p in (data_dir / "slipping").glob("*") if p.suffix.lower() in exts)
    stable_paths = sorted(p for p in (data_dir / "stable").glob("*") if p.suffix.lower() in exts)

    if not slip_paths or not stable_paths:
        raise FileNotFoundError(
            f"Require {data_dir}/slipping/ and {data_dir}/stable/ with image files."
        )

    print(f"  slipping={len(slip_paths)}  stable={len(stable_paths)}")

    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    W, H = image_size[1], image_size[0]

    class _TactileDataset(Dataset):
        def __init__(self, slip, stable):
            self.samples = [(p, 1.0) for p in slip] + [(p, 0.0) for p in stable]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            arr = np.array(
                _Image.open(path).convert("RGB").resize((W, H)), dtype=np.float32
            ) / 255.0
            arr = (arr - imagenet_mean) / imagenet_std
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            return tensor, torch.tensor(label, dtype=torch.float32)

    full_ds = _TactileDataset(slip_paths, stable_paths)
    n_val = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    dev = torch.device(device)
    model = _build_cnn().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_acc = 0.0
    print(f"Training SlipDetectorCNN on {dev} | {n_train} train / {n_val} val")

    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(dev), labels.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(imgs).squeeze(1), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(dev), labels.to(dev)
                preds = torch.sigmoid(model(imgs).squeeze(1)) >= 0.5
                correct += (preds == labels.bool()).sum().item()
                total += len(labels)
        acc = correct / max(total, 1)
        print(f"  Epoch {epoch:3d}/{epochs} | val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), output_path)
            print(f"  ✓ New best acc={best_acc:.4f} → {output_path}")

    print(f"\nDone. Best val accuracy: {best_acc:.4f}")
