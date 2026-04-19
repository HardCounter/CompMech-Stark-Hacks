"""
Shape from Silhouette (SfS) — visual hull reconstruction for HEAPGrasp.

Algorithm
---------
A 3-D voxel grid is initialised as fully occupied. For every silhouette view:

  1. Each voxel is projected onto the image plane using the hemispherical camera model.
  2. Voxels whose projection falls *inside* the image but *outside* the silhouette
     are carved away (they cannot belong to the object).
  3. Voxels outside the image frustum are left untouched (conservative).

The surviving voxels form the visual hull — a tight upper bound on object shape.

Hemispherical camera model
--------------------------
The hand-eye camera orbits on a hemisphere of radius `camera_distance` centred on
the object.  Each viewpoint is described by two angles:

  phi   — azimuthal angle around the world Y-axis (= radians(angle_deg))
  theta — polar angle from the zenith (Y-axis); theta=0 is directly above,
           theta=pi/2 is horizontal (legacy turntable mode)

Camera position in world frame:
  C = (-r·sin(θ)·sin(φ),  r·cos(θ),  -r·sin(θ)·cos(φ))

Camera rotation (world → camera):
  right = ( cos(φ),            0,          -sin(φ)          )
  up    = ( cos(θ)·sin(φ),   sin(θ),    cos(θ)·cos(φ)    )
  fwd   = ( sin(θ)·sin(φ),  -cos(θ),    sin(θ)·cos(φ)    )
  R = stack([right, up, fwd])

Transform:  pts_cam = R @ (pts_world - C)

At theta=pi/2 this is algebraically identical to the old turntable formula
  pts_cam = Ry(-angle) @ pts_world + [0, 0, camera_distance]
so all existing callers are unaffected when thetas_rad is omitted.
"""

from __future__ import annotations

from typing import List, Optional

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


def _view_transform(
    angle_deg: float,
    theta: float,
    camera_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute (R, C) for a camera on the hemisphere.

    R : (3, 3) world-to-camera rotation  —  pts_cam = R @ (pts_world - C)
    C : (3,)   camera position in world frame

    angle_deg : azimuthal angle (degrees); object rotation ≡ camera orbit
    theta     : polar angle from zenith in radians (pi/2 = horizontal)
    """
    phi = np.radians(angle_deg)
    r = camera_distance

    C = np.array([
        -r * np.sin(theta) * np.sin(phi),
         r * np.cos(theta),
        -r * np.sin(theta) * np.cos(phi),
    ], dtype=np.float64)

    fwd = -C / r  # unit vector from camera toward world origin

    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(fwd, world_up)) > 0.9999:   # camera directly above/below
        world_up = np.array([0.0, 0.0, 1.0])

    right = np.cross(world_up, fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)

    R = np.stack([right, up, fwd], axis=0)     # (3, 3)
    return R, C


def estimate_grid_center(
    silhouettes: List[np.ndarray],
    camera_matrix: np.ndarray,
    camera_distance: float,
    thetas_rad: Optional[List[float]] = None,
    angles_deg: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Estimate the 3-D object centre by back-projecting each view's silhouette
    centroid through the full hemispherical camera model and averaging.

    For horizontal views (theta=pi/2, default) this is identical to the
    previous formula X_c=(u-cx)*d/fx, Y_c=(v-cy)*d/fy.

    thetas_rad : polar angle per view (radians).  None → pi/2 for all views.
    angles_deg : azimuthal angle per view (degrees).  None → 0 for all views.
    """
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    n_views = len(silhouettes)
    if thetas_rad is None:
        thetas_rad = [np.pi / 2.0] * n_views
    if angles_deg is None:
        angles_deg = [0.0] * n_views

    centers, n = [], 0
    for mask, angle_deg, theta in zip(silhouettes, angles_deg, thetas_rad):
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        u_avg = float(xs.mean())
        v_avg = float(ys.mean())

        # Back-project centroid to the point at camera-frame depth = camera_distance
        p_cam = np.array([
            (u_avg - cx) * camera_distance / fx,
            (v_avg - cy) * camera_distance / fy,
            camera_distance,
        ])
        R, C = _view_transform(angle_deg, theta, camera_distance)
        p_world = R.T @ p_cam + C          # inverse of pts_cam = R @ (pts - C)
        centers.append(p_world)
        n += 1

    if n == 0:
        return np.zeros(3)
    return np.mean(centers, axis=0)


def shape_from_silhouette(
    silhouettes: List[np.ndarray],
    angles_deg: List[float],
    camera_matrix: np.ndarray,
    volume_size: int = 64,
    object_diameter: float = 0.15,
    camera_distance: float = 0.40,
    grid_center: np.ndarray | None = None,
    thetas_rad: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Reconstruct the visual hull from multi-view silhouettes.

    Args:
        silhouettes:      bool masks (H, W) from extract_silhouettes()
        angles_deg:       azimuthal rotation angle for each view (degrees)
        camera_matrix:    3×3 intrinsic matrix K
        volume_size:      voxel grid side length V (grid = V³ voxels)
        object_diameter:  side length of reconstruction volume in metres
        camera_distance:  hemisphere radius in metres
        grid_center:      (3,) world-space centre of the voxel grid.
                          None = auto-detect from silhouette centroids.
        thetas_rad:       polar angle from zenith per view (radians).
                          None → pi/2 for every view (horizontal, legacy mode).

    Returns:
        (V, V, V) bool ndarray — True where voxel is part of visual hull
    """
    V = volume_size
    half = object_diameter / 2.0
    K = camera_matrix

    n_views = len(silhouettes)
    if thetas_rad is None:
        thetas_rad = [np.pi / 2.0] * n_views

    if grid_center is None:
        grid_center = estimate_grid_center(
            silhouettes, camera_matrix, camera_distance,
            thetas_rad=thetas_rad, angles_deg=list(angles_deg),
        )

    cx0, cy0, cz0 = float(grid_center[0]), float(grid_center[1]), float(grid_center[2])

    lin_x = np.linspace(cx0 - half, cx0 + half, V)
    lin_y = np.linspace(cy0 - half, cy0 + half, V)
    lin_z = np.linspace(cz0 - half, cz0 + half, V)
    xi, yi, zi = np.meshgrid(lin_x, lin_y, lin_z, indexing='ij')
    pts = np.stack([xi.ravel(), yi.ravel(), zi.ravel()], axis=0)  # (3, V³)

    occupied = np.ones(V ** 3, dtype=bool)

    for mask, angle_deg, theta in zip(silhouettes, angles_deg, thetas_rad):
        H, W = mask.shape
        R, C = _view_transform(angle_deg, theta, camera_distance)

        pts_cam = R @ (pts - C[:, None])   # (3, N)
        depth = pts_cam[2]
        in_front = depth > 1e-3

        proj = K @ pts_cam                 # (3, N)
        u = proj[0] / (depth + 1e-9)
        v = proj[1] / (depth + 1e-9)

        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)

        in_frame = in_front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

        in_sil = np.zeros(V ** 3, dtype=bool)
        in_sil[in_frame] = mask[vi[in_frame], ui[in_frame]]

        occupied &= ~(in_frame & ~in_sil)

    return occupied.reshape(V, V, V)


def voxels_to_pointcloud(
    voxels: np.ndarray,
    object_diameter: float = 0.15,
    grid_center: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return occupied voxel centres as an (N, 3) metric point cloud.

    grid_center: the world-space centre used when shape_from_silhouette was
                 called (default zeros = grid centred at world origin).
                 Pass the same value used during reconstruction so that the
                 returned coordinates are in the correct world frame.
    """
    V = voxels.shape[0]
    half = object_diameter / 2.0
    if grid_center is None:
        grid_center = np.zeros(3)
    cx0, cy0, cz0 = float(grid_center[0]), float(grid_center[1]), float(grid_center[2])
    lin_x = np.linspace(cx0 - half, cx0 + half, V)
    lin_y = np.linspace(cy0 - half, cy0 + half, V)
    lin_z = np.linspace(cz0 - half, cz0 + half, V)
    xi, yi, zi = np.where(voxels)
    return np.stack([lin_x[xi], lin_y[yi], lin_z[zi]], axis=1)


def reproject_hull(
    voxels: np.ndarray,
    angles_deg: List[float],
    camera_matrix: np.ndarray,
    object_diameter: float,
    camera_distance: float,
    image_shape: tuple,
    thetas_rad: Optional[List[float]] = None,
    grid_center: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    Project the visual hull back onto each view's image plane.

    For each input view, occupied voxel centres are projected through the
    same camera model used during reconstruction. The resulting binary
    masks should closely match the original silhouettes; discrepancies
    diagnose specific failure modes:

      hull >> original  → masks were too large (shadows, noisy BG subtraction)
      hull << original  → hull is under-carved (camera distance / FOV wrong,
                          too few views, or wrong polar angle θ)
      hull ≈ original   → reconstruction is geometrically consistent

    Returns
    -------
    List of (H, W) bool masks, one per view.
    """
    H, W = image_shape
    n_views = len(angles_deg)
    K = camera_matrix

    if thetas_rad is None:
        thetas_rad = [np.pi / 2.0] * n_views

    pts = voxels_to_pointcloud(voxels, object_diameter, grid_center)  # (N, 3)
    if len(pts) == 0:
        return [np.zeros((H, W), dtype=bool)] * n_views

    import cv2 as _cv2
    kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (3, 3))

    results = []
    for angle_deg, theta in zip(angles_deg, thetas_rad):
        R, C = _view_transform(angle_deg, theta, camera_distance)
        pts_cam = R @ (pts - C).T          # (3, N)
        depth   = pts_cam[2]
        in_front = depth > 1e-3

        proj = K @ pts_cam
        u = proj[0] / (depth + 1e-9)
        v = proj[1] / (depth + 1e-9)

        valid = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        canvas = np.zeros((H, W), dtype=np.uint8)
        if valid.any():
            ui = np.round(u[valid]).astype(np.int32)
            vi = np.round(v[valid]).astype(np.int32)
            canvas[vi, ui] = 255
            # Close gaps between sparse projected voxels
            canvas = _cv2.dilate(canvas, kernel, iterations=2)

        results.append(canvas > 0)

    return results
