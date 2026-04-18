"""
Shape from Silhouette (SfS) — visual hull reconstruction for HEAPGrasp.

Algorithm
---------
A 3-D voxel grid is initialised as fully occupied. For every silhouette view:

  1. Each voxel is projected onto the image plane using the turntable camera model.
  2. Voxels whose projection falls *inside* the image but *outside* the silhouette
     are carved away (they cannot belong to the object).
  3. Voxels outside the image frustum are left untouched (conservative).

The surviving voxels form the visual hull — a tight upper bound on object shape.

Turntable camera model
----------------------
Camera is fixed; the object rotates by `angle_deg` around the world Y-axis between
views. This is equivalent to the camera orbiting by `-angle_deg` around Y while the
object stays still. Camera sits at (0, 0, camera_distance) looking toward the origin.

  pts_cam = Ry(-angle) @ pts_world + [0, 0, camera_distance]
"""

from __future__ import annotations

from typing import List

import numpy as np


def default_camera_matrix(width: int = 640, height: int = 480, fov_deg: float = 62.2) -> np.ndarray:
    """
    Approximate 3×3 intrinsic matrix K for the Pi Camera Module 3
    (horizontal FOV ≈ 66°; v2 ≈ 62.2°). Tune fov_deg to your module.

    For accurate results, calibrate with a checkerboard via cv2.calibrateCamera.
    """
    fx = (width / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx, 0.0, cx],
                     [0.0, fx, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def shape_from_silhouette(
    silhouettes: List[np.ndarray],
    angles_deg: List[float],
    camera_matrix: np.ndarray,
    volume_size: int = 64,
    object_diameter: float = 0.15,
    camera_distance: float = 0.40,
) -> np.ndarray:
    """
    Reconstruct the visual hull from multi-view silhouettes.

    Args:
        silhouettes:      bool masks (H, W) from extract_silhouettes()
        angles_deg:       turntable rotation angle for each view (degrees)
        camera_matrix:    3×3 intrinsic matrix K
        volume_size:      voxel grid side length V (grid = V³ voxels)
        object_diameter:  side length of reconstruction volume in metres
        camera_distance:  camera–object-centre distance in metres

    Returns:
        (V, V, V) bool ndarray — True where voxel is part of visual hull
    """
    V = volume_size
    half = object_diameter / 2.0
    K = camera_matrix

    # World grid: X right, Y up, Z toward camera at angle=0
    lin = np.linspace(-half, half, V)
    xi, yi, zi = np.meshgrid(lin, lin, lin, indexing='ij')
    # pts shape: (3, V³)
    pts = np.stack([xi.ravel(), yi.ravel(), zi.ravel()], axis=0)

    occupied = np.ones(V ** 3, dtype=bool)

    for mask, angle_deg in zip(silhouettes, angles_deg):
        H, W = mask.shape
        theta = np.radians(-angle_deg)          # camera orbit = negative object rotation

        c, s = np.cos(theta), np.sin(theta)
        Ry = np.array([[c, 0.0, s],
                       [0.0, 1.0, 0.0],
                       [-s, 0.0, c]], dtype=np.float64)

        pts_cam = Ry @ pts                      # (3, N)
        pts_cam[2] += camera_distance           # translate camera along +Z

        depth = pts_cam[2]
        in_front = depth > 1e-3

        # Perspective projection
        proj = K @ pts_cam                      # (3, N)
        u = proj[0] / (depth + 1e-9)
        v = proj[1] / (depth + 1e-9)

        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)

        in_frame = in_front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

        # Look up silhouette for in-frame voxels
        in_sil = np.zeros(V ** 3, dtype=bool)
        in_sil[in_frame] = mask[vi[in_frame], ui[in_frame]]

        # Carve voxels visible but outside silhouette
        occupied &= ~(in_frame & ~in_sil)

    return occupied.reshape(V, V, V)


def voxels_to_pointcloud(voxels: np.ndarray, object_diameter: float = 0.15) -> np.ndarray:
    """Return occupied voxel centres as an (N, 3) metric point cloud."""
    V = voxels.shape[0]
    lin = np.linspace(-object_diameter / 2.0, object_diameter / 2.0, V)
    xi, yi, zi = np.where(voxels)
    return np.stack([lin[xi], lin[yi], lin[zi]], axis=1)
