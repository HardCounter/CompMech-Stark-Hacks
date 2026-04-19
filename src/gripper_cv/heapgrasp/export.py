"""
Export reconstruction outputs to disk.

PLY files can be opened in MeshLab, CloudCompare, or Open3D.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


def save_ply(points: np.ndarray, path: str | Path, colors: Optional[np.ndarray] = None) -> None:
    """Write an ASCII PLY point cloud. colors should be (N, 3) uint8 RGB if provided."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    N = len(points)
    has_color = colors is not None and colors.shape == (N, 3)

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, pt in enumerate(points):
            row = f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}"
            if has_color:
                r, g, b = colors[i].astype(int) # type: ignore
                row += f" {r} {g} {b}"
            f.write(row + "\n")

    print(f"Saved {N} points → {path}")


def save_masks(masks: List[np.ndarray], output_dir: str | Path) -> None:
    """Save silhouette masks as PNG files for visual inspection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, mask in enumerate(masks):
        cv2.imwrite(str(output_dir / f"mask_{i:03d}.png"), mask.astype(np.uint8) * 255)
    print(f"Saved {len(masks)} masks → {output_dir}")


def save_voxels_npy(voxels: np.ndarray, path: str | Path) -> None:
    """Save raw voxel grid as .npy for further processing in Python."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, voxels)
    print(f"Saved voxel grid {voxels.shape} -> {path}")


def save_grasps_json(payload: dict, path: str | Path) -> None:
    """Save a grasps payload (from ``grasps_to_json``) as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {len(payload.get('grasps', []))} grasps -> {path}")
