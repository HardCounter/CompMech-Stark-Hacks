"""
Grasp candidate generation from a visual-hull point cloud.

This module takes the metric point cloud produced by
:func:`gripper_cv.heapgrasp.reconstruct.voxels_to_pointcloud` and returns a
ranked list of parallel-jaw grasps the arm can attempt.

Two families of scorers are exposed:

* :func:`pca_grasp`       — one shot, PCA-aligned (fast, deterministic).
* :func:`sample_grasps`   — many antipodal candidates around the PCA frame,
                            scored with simple geometric heuristics.

A dedicated :class:`LearnedGraspScorer` (ONNX/Hailo, GQ-CNN-style) is provided
in :mod:`gripper_cv.heapgrasp.grasp_learned` and can be plugged into
:func:`sample_grasps` via the ``score_fn`` hook.

Frame convention
----------------
All positions/directions use the same world frame as
:func:`shape_from_silhouette`:

    X  right    (turntable radius direction at angle 0)
    Y  up       (turntable rotation axis)
    Z  toward the camera at angle 0

and the origin is the turntable centre.  The camera sits at
``(0, 0, camera_distance)`` looking back toward the origin.

A :class:`GraspCandidate` describes a parallel-jaw grasp:

* ``position``  — midpoint between the two jaws (m).
* ``approach``  — unit vector the gripper moves **along** to reach the object.
* ``jaw_axis``  — unit vector along which the jaws open/close.
* ``width``     — jaw opening in metres, already including a safety clearance.
* ``score``     — quality heuristic in [0, 1]; higher is better.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]
ScoreFn = Callable[["GraspCandidate", np.ndarray], float]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraspCandidate:
    position: Vec3
    approach: Vec3
    jaw_axis: Vec3
    width: float
    score: float

    def to_dict(self) -> dict:
        """JSON-serialisable dict using SI units and short keys."""
        return {
            "position_m": [float(v) for v in self.position],
            "approach": [float(v) for v in self.approach],
            "jaw_axis": [float(v) for v in self.jaw_axis],
            "width_m": float(self.width),
            "score": float(self.score),
        }


@dataclass(frozen=True)
class ObjectShape:
    """PCA-based shape summary of the point cloud (world frame)."""

    centroid: Vec3
    # Principal axes, longest first (unit vectors, columns of eigvec matrix)
    long_axis: Vec3
    mid_axis: Vec3
    short_axis: Vec3
    # Physical extent along each axis (2σ diameter, metres)
    extents_m: Tuple[float, float, float]

    @classmethod
    def from_points(cls, points: np.ndarray) -> "ObjectShape":
        if len(points) < 4:
            raise ValueError("Need at least 4 points to estimate shape.")
        centroid, eigvals, eigvecs = _pca(points)
        spans = tuple(float(2.0 * np.sqrt(max(ev, 0.0)) * 2.5) for ev in eigvals)
        return cls(
            centroid=_totuple(centroid),
            long_axis=_totuple(eigvecs[:, 0]),
            mid_axis=_totuple(eigvecs[:, 1]),
            short_axis=_totuple(eigvecs[:, 2]),
            extents_m=spans,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Core PCA grasp
# ---------------------------------------------------------------------------

DEFAULT_CLEARANCE_M = 0.015
DEFAULT_MIN_WIDTH_M = 0.005
DEFAULT_MAX_WIDTH_M = 0.120


def pca_grasp(
    points: np.ndarray,
    clearance_m: float = DEFAULT_CLEARANCE_M,
    max_width_m: float = DEFAULT_MAX_WIDTH_M,
) -> GraspCandidate:
    """
    Build a single grasp centred on the point-cloud centroid using its PCA axes.

    Jaws open along the **longest** principal axis — so they close around the
    narrower cross-section — and the gripper approaches along the **shortest**
    axis, which is typically the flattest direction and has the shortest
    travel to contact.
    """
    shape = ObjectShape.from_points(points)
    jaw_axis = np.asarray(shape.long_axis, dtype=float)
    # Approach points *into* the object from outside; centroid - short_axis * r
    approach = -np.asarray(shape.short_axis, dtype=float)

    width = _clamped_width(shape.extents_m[0], clearance_m, max_width_m)
    score = _pca_score(shape, width, clearance_m, max_width_m)

    return GraspCandidate(
        position=shape.centroid,
        approach=_totuple(_unit(approach)),
        jaw_axis=_totuple(_unit(jaw_axis)),
        width=float(width),
        score=float(score),
    )


# ---------------------------------------------------------------------------
# Antipodal sampling — cheap candidate set around the PCA frame
# ---------------------------------------------------------------------------

def sample_grasps(
    points: np.ndarray,
    n_candidates: int = 64,
    clearance_m: float = DEFAULT_CLEARANCE_M,
    min_width_m: float = DEFAULT_MIN_WIDTH_M,
    max_width_m: float = DEFAULT_MAX_WIDTH_M,
    score_fn: Optional[ScoreFn] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[GraspCandidate]:
    """
    Generate a diverse set of parallel-jaw grasp candidates.

    Procedure:

    1. PCA the cloud to get the object frame (long / mid / short axes).
    2. Sample ``n_candidates`` lines parallel to the **long** axis at random
       offsets in the (mid, short) plane.
    3. For each line, project every point onto it, keep points whose lateral
       distance is under the gripper jaw thickness budget; the hit extent gives
       the jaw midpoint and a width estimate.
    4. Score with ``score_fn`` (defaults to :func:`default_score`).

    The longest axis is preferred because parallel-jaw grippers grasp across
    the **narrow** side of the object.
    """
    if len(points) < 4:
        raise ValueError("Need at least 4 points for antipodal sampling.")

    rng = rng or np.random.default_rng(0)
    shape = ObjectShape.from_points(points)
    centroid = np.asarray(shape.centroid, dtype=float)
    long_ax = np.asarray(shape.long_axis, dtype=float)
    mid_ax = np.asarray(shape.mid_axis, dtype=float)
    short_ax = np.asarray(shape.short_axis, dtype=float)

    # Offsets in the plane perpendicular to the jaw axis
    sigma_mid = max(shape.extents_m[1] / 5.0, 1e-3)
    sigma_short = max(shape.extents_m[2] / 5.0, 1e-3)
    u_offsets = rng.normal(0.0, sigma_mid, size=n_candidates)
    v_offsets = rng.normal(0.0, sigma_short, size=n_candidates)

    # Project cloud into the object frame, shape (N, 3): (long, mid, short)
    rel = points - centroid
    proj_long = rel @ long_ax
    proj_mid = rel @ mid_ax
    proj_short = rel @ short_ax

    # Points close enough to a candidate line contribute to its jaw extent
    jaw_thickness = max(clearance_m, 0.005)

    scorer = score_fn or default_score

    candidates: List[GraspCandidate] = []
    for u, v in zip(u_offsets, v_offsets):
        near = (np.abs(proj_mid - u) < jaw_thickness) & (
            np.abs(proj_short - v) < jaw_thickness
        )
        if near.sum() < 4:
            continue

        span = float(proj_long[near].max() - proj_long[near].min())
        if span < min_width_m:
            continue

        width = _clamped_width(span, clearance_m, max_width_m)
        midpoint_long = 0.5 * float(
            proj_long[near].max() + proj_long[near].min()
        )
        position = (
            centroid
            + midpoint_long * long_ax
            + u * mid_ax
            + v * short_ax
        )
        # Approach: from the camera side unless the jaw axis is already vertical
        approach = -short_ax if abs(short_ax[1]) < 0.9 else -mid_ax

        cand = GraspCandidate(
            position=_totuple(position),
            approach=_totuple(_unit(approach)),
            jaw_axis=_totuple(_unit(long_ax)),
            width=float(width),
            score=0.0,
        )
        s = float(scorer(cand, points))
        candidates.append(
            GraspCandidate(
                position=cand.position,
                approach=cand.approach,
                jaw_axis=cand.jaw_axis,
                width=cand.width,
                score=s,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def default_score(cand: GraspCandidate, points: np.ndarray) -> float:
    """
    Heuristic quality score in [0, 1].

    Rewards grasps that are:

    * close to the point cloud centroid (stable moment arm);
    * have a width comfortably below ``max_width_m`` but above ``min_width_m``;
    * centred in the cloud's long axis (symmetric contact).
    """
    centroid = points.mean(0)
    pos = np.asarray(cand.position, dtype=float)

    dist = float(np.linalg.norm(pos - centroid))
    extent = float(np.linalg.norm(points.max(0) - points.min(0)))
    proximity = max(0.0, 1.0 - dist / (0.5 * extent + 1e-6))

    w = cand.width
    width_fit = 1.0 - abs(
        (w - DEFAULT_MAX_WIDTH_M / 2.0) / (DEFAULT_MAX_WIDTH_M / 2.0 + 1e-6)
    )
    width_fit = max(0.0, min(1.0, width_fit))

    return float(0.65 * proximity + 0.35 * width_fit)


# ---------------------------------------------------------------------------
# High-level convenience wrappers
# ---------------------------------------------------------------------------

def grasp_from_voxels(
    voxels: np.ndarray,
    object_diameter: float,
    top_k: int = 1,
    n_candidates: int = 64,
    score_fn: Optional[ScoreFn] = None,
) -> List[GraspCandidate]:
    """
    Sample and rank grasps directly from an SfS occupancy grid.

    Returns ``top_k`` candidates ordered by score (highest first).  The first
    candidate is always the PCA grasp so downstream callers get a sensible
    deterministic fallback even if sampling fails (e.g. point cloud too small).
    """
    from .reconstruct import voxels_to_pointcloud

    points = voxels_to_pointcloud(voxels, object_diameter)
    if len(points) < 8:
        return []

    out: List[GraspCandidate] = [pca_grasp(points)]
    out.extend(
        sample_grasps(points, n_candidates=n_candidates, score_fn=score_fn)
    )

    # Drop near-duplicates (same position within 2 mm and same jaw axis within 10°)
    deduped = _dedupe(out, min_pos_dist_m=0.002, min_angle_deg=10.0)
    return deduped[:top_k]


def grasps_to_json(
    grasps: Sequence[GraspCandidate],
    *,
    frame: str = "turntable_world",
    extra: Optional[dict] = None,
) -> dict:
    """
    Package a list of grasps as a JSON-ready dict for the robot controller.

    The ``frame`` field is recorded explicitly so downstream transforms are
    unambiguous; see the module docstring for the convention.
    """
    doc = {
        "frame": frame,
        "convention": {
            "x": "right", "y": "up", "z": "toward_camera_at_angle_0",
            "angle_deg_sign": "negative_camera_orbit_around_y",
        },
        "grasps": [g.to_dict() for g in grasps],
    }
    if extra:
        doc["meta"] = extra
    return doc


# ---------------------------------------------------------------------------
# Natural-language plan (shared with the dashboard)
# ---------------------------------------------------------------------------

def format_grasp_plan(
    voxels: np.ndarray,
    object_diameter: float,
    camera_distance: float,
    n_views: int,
) -> str:
    """
    Render a human-readable arm movement plan.

    The plan interprets the grasp in **camera/gripper-tip frame** because the
    camera is mounted at the TCP.  This is the same layout the Streamlit
    dashboard used, factored out here so both the CLI and the dashboard render
    the same text.
    """
    from .reconstruct import voxels_to_pointcloud

    points = voxels_to_pointcloud(voxels, object_diameter)
    if len(points) < 8:
        return (
            "Insufficient point cloud data.\n"
            "Capture more views for a reliable grasp plan."
        )

    shape = ObjectShape.from_points(points)
    centroid = np.asarray(shape.centroid)
    long_ax = np.asarray(shape.long_axis)
    short_ax = np.asarray(shape.short_axis)
    spans_m = sorted(shape.extents_m, reverse=True)

    dim_str = "  x  ".join(f"{d * 100:.1f} cm" for d in spans_m)
    dx_cm = float(centroid[0]) * 100.0
    dy_cm = float(centroid[1]) * 100.0

    jaw_xy = np.array([long_ax[0], long_ax[1]])
    wrist_deg = (
        float(np.degrees(np.arctan2(jaw_xy[1], jaw_xy[0])))
        if np.linalg.norm(jaw_xy) > 0.1
        else 0.0
    )

    jaw_span_cm = float(2.0 * np.sqrt(max((spans_m[0] / 5.0) ** 2, 0.0))) * 100.0
    jaw_open_cm = jaw_span_cm + 1.5

    _axis_desc = {
        0: "horizontal (left-right)",
        1: "vertical (up-down)",
        2: "axial (depth)",
    }
    jaw_desc = _axis_desc[int(np.argmax(np.abs(long_ax)))]
    grasp_desc = _axis_desc[int(np.argmax(np.abs(short_ax)))]

    if spans_m[0] > 0.10:
        grip_type = "two-finger power grip (large object)"
    elif spans_m[0] < 0.04:
        grip_type = "precision pinch (small object)"
    else:
        grip_type = "standard parallel-jaw pinch"

    fill_pct = 100.0 * int(voxels.sum()) / voxels.size
    width = 65
    sep = "=" * width
    sub = "-" * (width - 4)

    lines: List[str] = [
        sep,
        "  THEORETICAL GRASP PLAN  -  Ford Industrial Arm",
        f"  {n_views}-view HEAPGrasp reconstruction  |  "
        f"camera distance {camera_distance * 100:.0f} cm",
        sep,
        "",
        "  OBJECT SUMMARY",
        f"  {'Estimated dimensions':<28}{dim_str}",
        f"  {'Jaw span axis':<28}{jaw_desc}",
        f"  {'Approach axis':<28}{grasp_desc}",
        f"  {'Recommended grip':<28}{grip_type}",
        f"  {'Reconstruction fill':<28}{fill_pct:.0f} %  "
        f"({int(voxels.sum()):,} occupied voxels)",
        "",
        f"  {sub}",
        "  1. LATERAL ALIGNMENT",
        f"  {sub}",
        "  Camera (mounted at gripper tip) sees the object at:",
    ]

    if abs(dx_cm) > 0.3:
        dirn = "right" if dx_cm > 0 else "left"
        axis = "right (+X)" if dx_cm > 0 else "left (-X)"
        lines.append(f"    - {abs(dx_cm):.1f} cm to the {dirn} of optical centre")
        lines.append(f"      -> Translate TCP {abs(dx_cm):.1f} cm {axis}")
    else:
        lines.append("    - Centred horizontally  (no X correction needed)")

    if abs(dy_cm) > 0.3:
        dirn = "above" if dy_cm > 0 else "below"
        axis = "up (+Y)" if dy_cm > 0 else "down (-Y)"
        lines.append(f"    - {abs(dy_cm):.1f} cm {dirn} optical centre")
        lines.append(f"      -> Translate TCP {abs(dy_cm):.1f} cm {axis}")
    else:
        lines.append("    - Centred vertically    (no Y correction needed)")

    lines += [
        "",
        f"  {sub}",
        "  2. WRIST ROTATION",
        f"  {sub}",
    ]
    if abs(wrist_deg) > 5.0:
        dirn = "counter-clockwise" if wrist_deg > 0 else "clockwise"
        lines += [
            f"  Jaw axis is {abs(wrist_deg):.0f} deg from horizontal in the image plane.",
            f"    -> Rotate wrist {abs(wrist_deg):.0f} deg  {dirn}  "
            "(align jaws with object long axis)",
        ]
    else:
        lines.append(
            "  Jaw axis is approximately horizontal - no wrist rotation needed."
        )

    lines += [
        "",
        f"  {sub}",
        "  3. OPEN JAWS",
        f"  {sub}",
        f"  Object span along jaw axis:  {jaw_span_cm:.1f} cm",
        f"    -> Set jaw opening to  {jaw_open_cm:.1f} cm",
        f"       ({jaw_span_cm:.1f} cm object  +  1.5 cm clearance)",
        "",
        f"  {sub}",
        "  4. APPROACH",
        f"  {sub}",
        f"  Object is {camera_distance * 100:.0f} cm from the gripper tip "
        "(along optical axis).",
        f"    -> Advance {camera_distance * 100:.0f} cm along gripper Z-axis.",
        "    -> Speed: SLOW  -  transparent/specular surface, uncertain contact.",
        "",
        f"  {sub}",
        "  5. GRASP",
        f"  {sub}",
        "    -> Close jaws to contact resistance.",
        "    -> Target force: LIGHT.  Do not over-squeeze transparent objects.",
        "    -> Monitor SlipDetectorCNN output:",
        "         stable     -> continue",
        "         slipping   -> pause, re-seat object, or abort mission.",
        "",
        f"  {sub}",
        "  6. RETRIEVE",
        f"  {sub}",
        "    -> Raise TCP 5 cm vertically to clear the turntable surface.",
        "    -> Translate to deposit location at REDUCED speed.",
        "    -> Confirm stable hold before increasing speed.",
        "",
        sep,
        "  SAFETY NOTES",
        "",
        "  1. This plan is THEORETICAL - generated from camera data only.",
        "  2. No real arm commands have been issued.",
        "  3. A human operator must verify all clearances before executing.",
        "  4. Reconstruction accuracy improves with more views.",
        f"     Current quality: {fill_pct:.0f}% voxel fill from {n_views} views.",
        sep,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pca(points: np.ndarray):
    centroid = points.mean(0)
    cov = np.cov((points - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return centroid, eigvals[order], eigvecs[:, order]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-9 else v / n


def _totuple(v: np.ndarray) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


def _clamped_width(raw_span_m: float, clearance_m: float, max_width_m: float) -> float:
    return float(min(max(raw_span_m, 0.0) + clearance_m, max_width_m))


def _pca_score(
    shape: ObjectShape,
    width: float,
    clearance_m: float,
    max_width_m: float,
) -> float:
    # Favour graspable widths and elongated (anisotropic) objects.
    long_m, mid_m, short_m = shape.extents_m
    anisotropy = 1.0 - short_m / (long_m + 1e-9)
    width_headroom = max(0.0, 1.0 - width / max_width_m)
    return float(0.6 * max(0.0, min(1.0, anisotropy)) + 0.4 * width_headroom)


def _dedupe(
    grasps: Sequence[GraspCandidate],
    min_pos_dist_m: float = 0.002,
    min_angle_deg: float = 10.0,
) -> List[GraspCandidate]:
    keep: List[GraspCandidate] = []
    min_cos = float(np.cos(np.radians(min_angle_deg)))
    for g in sorted(grasps, key=lambda c: c.score, reverse=True):
        pos = np.asarray(g.position)
        jaw = np.asarray(g.jaw_axis)
        duplicate = False
        for k in keep:
            d = float(np.linalg.norm(pos - np.asarray(k.position)))
            cos = float(abs(np.dot(jaw, np.asarray(k.jaw_axis))))
            if d < min_pos_dist_m and cos > min_cos:
                duplicate = True
                break
        if not duplicate:
            keep.append(g)
    return keep


__all__ = [
    "GraspCandidate",
    "ObjectShape",
    "pca_grasp",
    "sample_grasps",
    "default_score",
    "grasp_from_voxels",
    "grasps_to_json",
    "format_grasp_plan",
]
