"""
Domain randomization for sim-to-real transfer.

Two use cases:
  Training:   DomainRandomTransform — augments (image, mask) pairs during training.
  Inference:  BackgroundRandomizer  — replaces background of a live segmented frame.

Both operate on (image: np.ndarray HxWx3 uint8, mask: np.ndarray HxW) pairs.
The mask can be bool (True=foreground) or int64 class-index (0=background);
background is always identified by mask == 0.

No torch import anywhere in this module.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Background texture generation
# ---------------------------------------------------------------------------

def random_background(
    size: Tuple[int, int],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a random background texture of shape (H, W, 3) uint8.

    Three styles chosen at random:
      0 — solid colour
      1 — horizontal gradient between two random colours
      2 — structured RGB noise

    Args:
        size: (H, W) in pixels
        rng:  numpy Generator (fresh default_rng if None)
    """
    if rng is None:
        rng = np.random.default_rng()
    H, W = size
    style = int(rng.integers(0, 3))
    img = np.zeros((H, W, 3), dtype=np.uint8)

    if style == 0:
        color = rng.integers(20, 220, size=3).tolist()
        img[:] = color

    elif style == 1:
        c1 = rng.integers(20, 200, size=3).astype(np.float32)
        c2 = rng.integers(20, 200, size=3).astype(np.float32)
        t = np.linspace(0.0, 1.0, W, dtype=np.float32)
        for ch in range(3):
            row = (c1[ch] * (1.0 - t) + c2[ch] * t).astype(np.uint8)
            img[:, :, ch] = row[None, :]

    else:
        base = rng.integers(30, 180, size=(1, 1, 3)).astype(np.float32)
        noise = rng.normal(0.0, 25.0, size=(H, W, 3))
        img[:] = np.clip(base + noise, 0, 255).astype(np.uint8)

    return img


# ---------------------------------------------------------------------------
# BackgroundRandomizer — inference-time background replacement
# ---------------------------------------------------------------------------

class BackgroundRandomizer:
    """
    Replaces background pixels in a live RGB frame at inference time.

    If bg_dirs are given, textures are sampled uniformly at random from
    all image files found under those directories.  Falls back to
    random_background() if no files are found or bg_dirs is None.

    Args:
        bg_dirs:  directories containing .jpg/.png background textures
        rng:      numpy Generator (created fresh if None)
    """

    def __init__(
        self,
        bg_dirs: Sequence[Path] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._rng = rng or np.random.default_rng()
        self._paths: list[Path] = []
        if bg_dirs:
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            for d in bg_dirs:
                self._paths.extend(
                    p for p in Path(d).iterdir() if p.suffix.lower() in exts
                )

    def __call__(self, frame_rgb: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
        """
        Replace pixels where fg_mask is 0/False with a random background.

        Args:
            frame_rgb: (H, W, 3) uint8
            fg_mask:   (H, W) bool or int; 0/False = background

        Returns:
            (H, W, 3) uint8 with background replaced
        """
        H, W = frame_rgb.shape[:2]

        if self._paths:
            idx = int(self._rng.integers(0, len(self._paths)))
            from PIL import Image
            bg = np.array(
                Image.open(self._paths[idx]).convert("RGB").resize((W, H))
            )
        else:
            bg = random_background((H, W), self._rng)

        result = frame_rgb.copy()
        bg_pixels = ~fg_mask.astype(bool)
        result[bg_pixels] = bg[bg_pixels]
        return result


# ---------------------------------------------------------------------------
# DomainRandomTransform — training-time augmentation chain
# ---------------------------------------------------------------------------

class DomainRandomTransform:
    """
    Chain of domain-randomization augmentations for (image, mask) training pairs.

    Applied operations (each with its own probability):
      1. Background replacement — replaces pixels where mask==0 with random texture
      2. Colour jitter on foreground pixels — ±delta per RGB channel
      3. Gaussian noise — added to the whole image
      4. Random horizontal flip

    Input/output contract:
        image : (H, W, 3) uint8 numpy
        mask  : (H, W) int64 or bool — 0/False = background

    Args:
        bg_dirs:           directories with texture images (optional)
        p_bg_replace:      probability of background replacement (default 0.6)
        p_color_jitter:    probability of foreground colour jitter (default 0.8)
        color_jitter_range: ± pixel value range for colour jitter (default 40)
        p_noise:           probability of Gaussian noise (default 0.4)
        noise_std:         Gaussian sigma in pixel units (default 10)
        p_flip:            probability of horizontal flip (default 0.5)
    """

    def __init__(
        self,
        bg_dirs: Sequence[Path] | None = None,
        p_bg_replace: float = 0.6,
        p_color_jitter: float = 0.8,
        color_jitter_range: int = 40,
        p_noise: float = 0.4,
        noise_std: float = 10.0,
        p_flip: float = 0.5,
    ) -> None:
        self._randomizer = BackgroundRandomizer(bg_dirs)
        self.p_bg_replace = p_bg_replace
        self.p_color_jitter = p_color_jitter
        self.color_jitter_range = color_jitter_range
        self.p_noise = p_noise
        self.noise_std = noise_std
        self.p_flip = p_flip

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng()

        # 1. Background replacement
        if random.random() < self.p_bg_replace:
            image = self._randomizer(image, mask)

        # 2. Colour jitter on foreground pixels
        if random.random() < self.p_color_jitter:
            fg = mask.astype(bool)
            if fg.any():
                image = image.copy()
                delta = rng.integers(
                    -self.color_jitter_range, self.color_jitter_range + 1, size=3
                )
                image[fg] = np.clip(
                    image[fg].astype(np.int16) + delta, 0, 255
                ).astype(np.uint8)

        # 3. Gaussian noise
        if random.random() < self.p_noise:
            sigma = rng.uniform(self.noise_std * 0.5, self.noise_std * 2.0)
            noise = rng.normal(0.0, sigma, size=image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 4. Horizontal flip
        if random.random() < self.p_flip:
            image = image[:, ::-1].copy()
            mask = mask[:, ::-1].copy()

        return image, mask
