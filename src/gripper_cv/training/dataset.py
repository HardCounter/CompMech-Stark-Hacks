"""
Datasets for transparent-object segmentation fine-tuning.

Label conventions (4-class HEAPGrasp schema):
  0 = background
  1 = transparent object
  2 = specular (mirror-like) object
  3 = opaque object

Public API
----------
get_dataset_classes() → (TransposeDataset, SyntheticTransparentDataset)
    Lazily imports torch and returns the two Dataset subclasses.
    Results are cached after the first call.

split_dataset(dataset, val_fraction, seed) → (train_subset, val_subset)
    Lazily imports torch.utils.data.random_split.

Supported dataset formats
-------------------------
TransposeDataset: directory with images/ and masks/ subdirectories.
    images/  *.jpg or *.png (RGB)
    masks/   *.png (grayscale; pixel value = class index 0–3)
    Mask filename must share the same stem as the image file.

SyntheticTransparentDataset: generates labelled (image, mask) pairs on-the-fly.
    No disk data required. Useful for smoke-testing the training pipeline.
    Each sample is deterministic for its index (reproducible DataLoader workers).

Design note — lazy torch
------------------------
torch.utils.data.Dataset cannot be used as a base class without importing torch.
All class definitions are therefore deferred inside _get_dataset_classes(), which
is called only when training actually runs. This keeps the module importable on
machines without PyTorch installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

NUM_CLASSES: int = 4
CLASS_NAMES: Tuple[str, ...] = ("background", "transparent", "specular", "opaque")

_CLASSES_CACHE: Optional[Tuple[type, type]] = None


# ---------------------------------------------------------------------------
# Synthetic sample generation (pure numpy/cv2 — no torch)
# ---------------------------------------------------------------------------

def _synthetic_background(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    style = int(rng.integers(0, 3))
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if style == 0:
        img[:] = rng.integers(20, 220, size=3).tolist()
    elif style == 1:
        c1 = rng.integers(20, 200, size=3).astype(np.float32)
        c2 = rng.integers(20, 200, size=3).astype(np.float32)
        t = np.linspace(0.0, 1.0, W, dtype=np.float32)
        for ch in range(3):
            img[:, :, ch] = (c1[ch] * (1 - t) + c2[ch] * t).astype(np.uint8)[None, :]
    else:
        base = rng.integers(30, 180, size=(1, 1, 3)).astype(np.float32)
        img[:] = np.clip(base + rng.normal(0.0, 25.0, (H, W, 3)), 0, 255).astype(np.uint8)
    return img


def _generate_synthetic_sample(
    idx: int, H: int, W: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (H,W,3) uint8 image and (H,W) int64 mask. Deterministic per idx."""
    rng = np.random.default_rng(idx)
    img = _synthetic_background(H, W, rng)
    mask = np.zeros((H, W), dtype=np.int64)

    for _ in range(int(rng.integers(1, 5))):
        cls = int(rng.integers(1, 4))          # 1=transparent, 2=specular, 3=opaque
        ax = int(rng.integers(max(1, W // 10), max(2, W // 3)))
        ay = int(rng.integers(max(1, H // 10), max(2, H // 3)))
        cx = int(rng.integers(ax, W - ax))
        cy = int(rng.integers(ay, H - ay))
        ang = int(rng.integers(0, 180))

        if cls == 1:  # transparent: alpha blend
            overlay = img.copy()
            cv2.ellipse(overlay, (cx, cy), (ax, ay), ang, 0, 360,
                        rng.integers(80, 220, size=3).tolist(), -1)
            cv2.addWeighted(overlay, float(rng.uniform(0.2, 0.5)), img,
                            1.0 - float(rng.uniform(0.2, 0.5)), 0, img)
        elif cls == 2:  # specular: bright + highlight
            cv2.ellipse(img, (cx, cy), (ax, ay), ang, 0, 360,
                        rng.integers(160, 255, size=3).tolist(), -1)
            hx = max(0, cx - ax // 4)
            hy = max(0, cy - ay // 4)
            cv2.ellipse(img, (hx, hy),
                        (max(1, ax // 4), max(1, ay // 4)), ang, 0, 360,
                        (255, 255, 255), -1)
        else:  # opaque: solid
            cv2.ellipse(img, (cx, cy), (ax, ay), ang, 0, 360,
                        rng.integers(30, 180, size=3).tolist(), -1)

        region = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(region, (cx, cy), (ax, ay), ang, 0, 360, 1, -1)
        mask[region > 0] = cls

    return img, mask


# ---------------------------------------------------------------------------
# Lazy dataset class factory
# ---------------------------------------------------------------------------

def _get_dataset_classes() -> Tuple[type, type]:
    import torch
    from torch.utils.data import Dataset

    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    class TransposeDataset(Dataset):
        """
        Loads paired image/mask files from a Transpose-format directory.

        Directory layout:
            <root>/
                images/   *.jpg | *.png   (RGB)
                masks/    *.png            (grayscale, pixel = class 0–3)
            Mask filename must share the same stem as the image file.

        Args:
            root:      path to dataset root
            transform: DomainRandomTransform or None; applied to
                       (np_image uint8, np_mask int64) before tensor conversion
            img_size:  (H, W) target — both image and mask are resized to this
        """

        def __init__(
            self,
            root: str | Path,
            transform: Optional[Callable] = None,
            img_size: Tuple[int, int] = (512, 512),
        ) -> None:
            from PIL import Image  # noqa: F401 — verify import here
            root = Path(root)
            self.img_dir = root / "images"
            self.mask_dir = root / "masks"
            self.transform = transform
            self._H, self._W = img_size
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            self.img_paths = sorted(
                p for p in self.img_dir.iterdir() if p.suffix.lower() in exts
            )
            if not self.img_paths:
                raise FileNotFoundError(f"No images found in {self.img_dir}")

        def __len__(self) -> int:
            return len(self.img_paths)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            from PIL import Image
            img_path = self.img_paths[idx]
            mask_path = self.mask_dir / (img_path.stem + ".png")

            img = np.array(
                Image.open(img_path).convert("RGB").resize(
                    (self._W, self._H), Image.BILINEAR
                )
            )
            mask = np.array(
                Image.open(mask_path).convert("L").resize(
                    (self._W, self._H), Image.NEAREST
                )
            )
            mask = np.clip(mask, 0, NUM_CLASSES - 1).astype(np.int64)

            if self.transform is not None:
                img, mask = self.transform(img, mask)

            t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
            return t, torch.from_numpy(mask)

    class SyntheticTransparentDataset(Dataset):
        """
        Generates labelled training pairs on-the-fly — no disk dataset required.

        Each sample is deterministic for its index, so DataLoader workers produce
        reproducible outputs even under multiprocessing.

        Generation:
          • Background: random solid colour, gradient, or noise
          • 1–4 objects per image, each randomly typed:
              class 1 (transparent): alpha-blended ellipse
              class 2 (specular):    bright ellipse with specular highlight
              class 3 (opaque):      solid-colour ellipse

        Args:
            length:    virtual dataset length
            img_size:  (H, W) output size
            transform: DomainRandomTransform or None
            seed:      added to the per-sample rng seed for independent datasets
        """

        def __init__(
            self,
            length: int = 1000,
            img_size: Tuple[int, int] = (512, 512),
            transform: Optional[Callable] = None,
            seed: int = 0,
        ) -> None:
            self._length = length
            self._H, self._W = img_size
            self.transform = transform
            self._seed = seed

        def __len__(self) -> int:
            return self._length

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            img, mask = _generate_synthetic_sample(self._seed + idx, self._H, self._W)

            if self.transform is not None:
                img, mask = self.transform(img, mask)

            t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
            return t, torch.from_numpy(mask)

    return TransposeDataset, SyntheticTransparentDataset


def get_dataset_classes() -> Tuple[type, type]:
    """
    Lazily build and cache the torch Dataset subclasses.

    Returns:
        (TransposeDataset, SyntheticTransparentDataset)

    Raises:
        ImportError if torch is not installed.
    """
    global _CLASSES_CACHE
    if _CLASSES_CACHE is None:
        _CLASSES_CACHE = _get_dataset_classes()
    return _CLASSES_CACHE


def split_dataset(
    dataset,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple:
    """
    Split any torch Dataset into (train, val) subsets.

    Args:
        dataset:       any torch.utils.data.Dataset
        val_fraction:  fraction of samples used for validation
        seed:          RNG seed for reproducible splits

    Returns:
        (train_subset, val_subset)
    """
    import torch
    from torch.utils.data import random_split

    n = len(dataset)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    return random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
