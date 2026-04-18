"""
Auto-calibration for HEAPGrasp: determine camera_distance and object_diameter
from images rather than manual measurement.

Camera distance
---------------
An ArUco marker of known physical size is placed flat on the turntable. It is
visible in the background frame (object removed). cv2.solvePnP with
SOLVEPNP_IPPE_SQUARE gives the marker pose in the camera frame; the Euclidean
norm of the translation vector equals the camera-to-marker-centre distance,
which is the same camera_distance used by shape_from_silhouette.

Object diameter
---------------
After segmentation, each silhouette's bounding box is back-projected to metres
via the pinhole model:  W_m = w_px * Z / fx.  The maximum across all views
gives the tightest upper-bound on object extent, plus a 5% safety margin so
the voxel grid always fully encloses the object.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .capture import CaptureSession


@dataclass
class CalibrationResult:
    camera_distance: float
    object_diameter: float           # 0.0 until estimate_object_diameter() is called
    source: str                      # "aruco+silhouette", "silhouette_only", "manual"
    marker_id: Optional[int] = None
    tvec_raw: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# ArUco marker pose estimation
# ---------------------------------------------------------------------------

def detect_aruco_marker(
    image_rgb: np.ndarray,
    camera_matrix: np.ndarray,
    marker_size_m: float = 0.05,
    aruco_dict_id: int = cv2.aruco.DICT_4X4_50,
) -> tuple[np.ndarray | None, int | None]:
    """
    Detect the first ArUco marker in image_rgb and estimate its pose.

    Returns (tvec, marker_id) on success, (None, None) if no marker found.
    tvec is a (3,) float64 array in metres (camera frame).

    Uses SOLVEPNP_IPPE_SQUARE — the recommended replacement for the deprecated
    estimatePoseSingleMarkers in OpenCV 4.7+.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None, None

    # Use the first detected marker
    corners_0 = corners[0].reshape(4, 2).astype(np.float64)
    marker_id = int(ids[0][0])

    # 3-D corners in marker frame (Z=0 plane), clockwise from top-left.
    # This layout matches what ArucoDetector returns and what IPPE_SQUARE expects.
    half = marker_size_m / 2.0
    obj_pts = np.array(
        [[-half,  half, 0.0],
         [ half,  half, 0.0],
         [ half, -half, 0.0],
         [-half, -half, 0.0]],
        dtype=np.float64,
    )

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, corners_0,
        camera_matrix, None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not ok:
        return None, None

    return tvec.ravel(), marker_id


# ---------------------------------------------------------------------------
# Object diameter from silhouette bounding boxes
# ---------------------------------------------------------------------------

def estimate_object_diameter(
    masks: List[np.ndarray],
    camera_matrix: np.ndarray,
    camera_distance: float,
    padding: float = 1.05,
    fallback_diameter: float = 0.15,
) -> float:
    """
    Estimate the object's bounding diameter in metres from binary silhouettes.

    For each mask, the bounding box width and height (pixels) are converted to
    metres via the pinhole model:  W_m = w_px * Z / fx.  The maximum across all
    views gives a conservative upper bound, padded by `padding` (default 5%).
    """
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    max_dim = 0.0

    for mask in masks:
        mask_u8 = mask.astype(np.uint8)
        if not mask_u8.any():
            continue
        x, y, w, h = cv2.boundingRect(mask_u8)
        w_m = w * camera_distance / fx
        h_m = h * camera_distance / fy
        max_dim = max(max_dim, w_m, h_m)

    if max_dim == 0.0:
        warnings.warn(
            "All silhouette masks are empty; cannot estimate object diameter. "
            f"Falling back to {fallback_diameter} m.",
            UserWarning,
            stacklevel=2,
        )
        return fallback_diameter

    return max_dim * padding


# ---------------------------------------------------------------------------
# High-level auto-calibrate
# ---------------------------------------------------------------------------

def auto_calibrate(
    session: CaptureSession,
    camera_matrix: np.ndarray,
    marker_size_m: float = 0.05,
    aruco_dict_id: int = cv2.aruco.DICT_4X4_50,
    fallback_diameter: float = 0.15,
    fallback_distance: float = 0.40,
) -> CalibrationResult:
    """
    Detect an ArUco marker in the background frame and return camera_distance.
    object_diameter is left as 0.0 — call estimate_object_diameter() after
    segmentation and store the result in CalibrationResult.object_diameter.

    Falls back to manual defaults with a warning if detection fails.
    """
    if session.background is None:
        warnings.warn(
            "No background frame available for ArUco detection. "
            f"Using manual defaults (distance={fallback_distance} m, "
            f"diameter={fallback_diameter} m).",
            UserWarning,
            stacklevel=2,
        )
        return CalibrationResult(fallback_distance, fallback_diameter, "manual")

    tvec, marker_id = detect_aruco_marker(
        session.background, camera_matrix, marker_size_m, aruco_dict_id
    )

    if tvec is None:
        warnings.warn(
            "ArUco marker not detected in background frame. "
            f"Using manual defaults (distance={fallback_distance} m, "
            f"diameter={fallback_diameter} m). "
            "Ensure the marker is flat, fully visible, and well-lit.",
            UserWarning,
            stacklevel=2,
        )
        return CalibrationResult(fallback_distance, fallback_diameter, "manual")

    camera_distance = float(np.linalg.norm(tvec))
    print(f"  [calibrate] ArUco ID {marker_id} detected → camera distance: {camera_distance:.3f} m")

    return CalibrationResult(
        camera_distance=camera_distance,
        object_diameter=0.0,          # filled by pipeline after segmentation
        source="aruco",
        marker_id=marker_id,
        tvec_raw=tvec,
    )
