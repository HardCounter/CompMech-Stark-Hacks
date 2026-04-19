"""
Antipodal grasp planning on a Shape-from-Silhouette visual hull.

A parallel-jaw gripper grasp is defined by:
  - closing_dir : unit vector along which the jaws move toward each other
  - contact_1   : jaw-A surface contact point (world metres)
  - contact_2   : jaw-B surface contact point (world metres)
  - center       : TCP target = midpoint of contacts
  - approach_dir : direction the arm descends to reach the grasp position
  - width        : jaw opening in metres

Algorithm
---------
1. Extract outer-surface voxels (occupied voxels with at least one empty
   6-connected neighbour) and estimate their outward normals.
2. Sample N candidate closing directions from a hemisphere, plus the three
   PCA principal axes (the most likely stable closing directions).
3. For each direction d, find the two extreme surface points (antipodal pair)
   along d and score the grasp by:
     - normal_score  : surface normals at the contacts oppose the jaws
                       (friction-cone condition)
     - width_score   : Gaussian preference for an "ideal" jaw opening
     - center_score  : midpoint close to the object centroid (stability)
4. De-duplicate directions that are nearly parallel and return top-K.

Utility exports
---------------
grasp_from_voxels  — convenience wrapper around find_grasps
grasps_to_json     — package candidates as a JSON-ready dict
format_grasp_plan  — human-readable arm movement plan (shared with dashboard)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .reconstruct import voxels_to_pointcloud

Vec3 = Tuple[float, float, float]
ScoreFn = Callable[["GraspCandidate", np.ndarray], float]


# ---------------------------------------------------------------------------
# Primary grasp data model
# ---------------------------------------------------------------------------

@dataclass
class GraspCandidate:
    """One antipodal grasp candidate for a parallel-jaw gripper."""
    center: np.ndarray       # (3,) TCP target in world metres
    contact_1: np.ndarray    # (3,) jaw-A contact point
    contact_2: np.ndarray    # (3,) jaw-B contact point
    closing_dir: np.ndarray  # (3,) unit vector: c1 → c2 direction
    approach_dir: np.ndarray # (3,) unit vector: arm descends along this
    width: float             # jaw opening in metres
    score: float             # overall quality [0, 1]
    normal_score: float = 0.0
    width_score: float = 0.0
    center_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "center_m":      [float(v) for v in self.center],
            "contact_1_m":   [float(v) for v in self.contact_1],
            "contact_2_m":   [float(v) for v in self.contact_2],
            "closing_dir":   [float(v) for v in self.closing_dir],
            "approach_dir":  [float(v) for v in self.approach_dir],
            "width_m":       float(self.width),
            "score":         float(self.score),
            "normal_score":  float(self.normal_score),
            "width_score":   float(self.width_score),
            "center_score":  float(self.center_score),
        }


# ---------------------------------------------------------------------------
# PCA shape summary (used by format_grasp_plan)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectShape:
    """PCA-based shape summary of the point cloud (world frame)."""
    centroid: Vec3
    long_axis: Vec3
    mid_axis: Vec3
    short_axis: Vec3
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
# Antipodal planner internals
# ---------------------------------------------------------------------------

def _extract_surface(
    voxels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (indices, normals) for all surface voxels.

    A surface voxel is occupied with at least one empty 6-connected neighbour.
    The outward normal is the normalised sum of directions toward empty neighbours.
    """
    occ = voxels.astype(bool)
    is_surface = np.zeros(occ.shape, dtype=bool)
    normals = np.zeros(occ.shape + (3,), dtype=np.float32)

    for ax in range(3):
        for sign in (1, -1):
            nbr = np.roll(occ, sign, axis=ax)
            edge = [slice(None)] * 3
            edge[ax] = 0 if sign > 0 else -1
            nbr[tuple(edge)] = False

            cond = occ & ~nbr
            is_surface |= cond

            direction = np.zeros(3, dtype=np.float32)
            direction[ax] = float(sign)
            normals[..., 0] += cond * direction[0]
            normals[..., 1] += cond * direction[1]
            normals[..., 2] += cond * direction[2]

    si, sj, sk = np.where(is_surface & occ)
    if len(si) == 0:
        return np.empty((0, 3), dtype=int), np.empty((0, 3), dtype=np.float32)

    norms_out = normals[si, sj, sk]
    mag = np.linalg.norm(norms_out, axis=1, keepdims=True)
    norms_out /= np.maximum(mag, 1e-8)
    return np.stack([si, sj, sk], axis=1), norms_out


def _sample_hemisphere(n: int, seed: int = 42) -> np.ndarray:
    """Return (n, 3) unit vectors uniformly distributed over a hemisphere (z≥0)."""
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    cos_t = rng.uniform(0.0, 1.0, n)
    sin_t = np.sqrt(1.0 - cos_t ** 2)
    return np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)


def _approach_for(closing_dir: np.ndarray) -> np.ndarray:
    """
    Pick a feasible approach direction perpendicular to closing_dir.
    Biased toward world -Y (downward) for a hand-eye arm coming from above.
    Falls back to -Z if closing_dir is nearly vertical.
    """
    world_down = np.array([0.0, -1.0, 0.0])
    proj = world_down - np.dot(world_down, closing_dir) * closing_dir
    if np.linalg.norm(proj) < 0.1:
        proj = np.array([0.0, 0.0, -1.0])
        proj -= np.dot(proj, closing_dir) * closing_dir
    return proj / np.linalg.norm(proj)


# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------

def find_grasps(
    voxels: np.ndarray,
    object_diameter: float,
    grid_center: Optional[np.ndarray] = None,
    max_jaw_width: float = 0.08,
    n_directions: int = 256,
    top_k: int = 5,
    ideal_width_frac: float = 0.55,
) -> List[GraspCandidate]:
    """
    Find antipodal grasp candidates on a visual hull.

    Parameters
    ----------
    voxels          : (V, V, V) bool occupancy grid
    object_diameter : reconstruction volume side length in metres
    grid_center     : world-frame origin of the voxel grid
    max_jaw_width   : maximum gripper opening in metres  (default 80 mm)
    n_directions    : number of candidate closing directions to sample
    top_k           : candidates to return (sorted best-first)
    ideal_width_frac: preferred width as fraction of object_diameter
    """
    pts_all = voxels_to_pointcloud(voxels, object_diameter, grid_center)
    if len(pts_all) < 8:
        return []

    V = voxels.shape[0]
    ctr = np.zeros(3) if grid_center is None else np.asarray(grid_center, dtype=float)
    half = object_diameter / 2.0

    surf_idx, surf_normals = _extract_surface(voxels)
    if len(surf_idx) == 0:
        return []

    lin_x = np.linspace(ctr[0] - half, ctr[0] + half, V)
    lin_y = np.linspace(ctr[1] - half, ctr[1] + half, V)
    lin_z = np.linspace(ctr[2] - half, ctr[2] + half, V)
    surf_pts = np.stack([
        lin_x[surf_idx[:, 0]],
        lin_y[surf_idx[:, 1]],
        lin_z[surf_idx[:, 2]],
    ], axis=1)

    centroid = pts_all.mean(0)

    cov = np.cov((pts_all - centroid).T)
    _, eigvecs = np.linalg.eigh(cov)
    pca_dirs = np.vstack([eigvecs.T, -eigvecs.T])
    hem_dirs = _sample_hemisphere(n_directions)
    all_dirs = np.vstack([pca_dirs, hem_dirs])

    ideal_width = ideal_width_frac * object_diameter
    candidates: List[GraspCandidate] = []
    seen_dirs: List[np.ndarray] = []

    for d_raw in all_dirs:
        norm = np.linalg.norm(d_raw)
        if norm < 1e-8:
            continue
        d = d_raw / norm

        projs = surf_pts @ d
        i1, i2 = int(np.argmin(projs)), int(np.argmax(projs))
        width = float(projs[i2] - projs[i1])

        if width < 0.003 or width > max_jaw_width:
            continue

        if any(abs(float(np.dot(d, s))) > 0.95 for s in seen_dirs):
            continue

        c1 = surf_pts[i1]
        c2 = surf_pts[i2]
        n1 = surf_normals[i1]
        n2 = surf_normals[i2]

        f1 = float(np.dot(n1, -d))
        f2 = float(np.dot(n2,  d))
        normal_score = float(np.clip((f1 + f2) / 2.0, 0.0, 1.0))

        width_score = float(
            np.exp(-((width - ideal_width) / (ideal_width * 0.5)) ** 2)
        )

        center = (c1 + c2) / 2.0
        center_score = float(
            np.exp(-np.linalg.norm(center - centroid) / (object_diameter * 0.3))
        )

        score = 0.35 * normal_score + 0.40 * width_score + 0.25 * center_score

        candidates.append(GraspCandidate(
            center=center,
            contact_1=c1,
            contact_2=c2,
            closing_dir=d.copy(),
            approach_dir=_approach_for(d),
            width=width,
            score=score,
            normal_score=normal_score,
            width_score=width_score,
            center_score=center_score,
        ))
        seen_dirs.append(d)

    candidates.sort(key=lambda g: -g.score)
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def grasp_from_voxels(
    voxels: np.ndarray,
    object_diameter: float,
    top_k: int = 5,
    n_candidates: int = 256,
    score_fn: Optional[ScoreFn] = None,
) -> List[GraspCandidate]:
    """
    Sample and rank grasps directly from an SfS occupancy grid.

    ``score_fn`` is accepted for API compatibility with the learned scorer
    (``LearnedGraspScorer.score_fn``) but is not used by the geometric planner;
    pass it only when you have a trained ONNX/Hailo scorer available.
    """
    return find_grasps(voxels, object_diameter, n_directions=n_candidates, top_k=top_k)


def grasps_to_json(
    grasps: Sequence[GraspCandidate],
    *,
    frame: str = "turntable_world",
    extra: Optional[dict] = None,
) -> dict:
    """Package a list of grasps as a JSON-ready dict for the robot controller."""
    doc: dict = {
        "frame": frame,
        "convention": {
            "x": "right", "y": "up", "z": "toward_camera_at_angle_0",
        },
        "grasps": [g.to_dict() for g in grasps],
    }
    if extra:
        doc["meta"] = extra
    return doc


# ---------------------------------------------------------------------------
# Natural-language plan (shared with dashboard)
# ---------------------------------------------------------------------------

def format_grasp_plan(
    voxels: np.ndarray,
    object_diameter: float,
    camera_distance: float,
    n_views: int,
) -> str:
    """Render a human-readable arm movement plan from the visual hull."""
    points = voxels_to_pointcloud(voxels, object_diameter)
    if len(points) < 8:
        return (
            "Insufficient point cloud data.\n"
            "Capture more views for a reliable grasp plan."
        )

    shape = ObjectShape.from_points(points)
    centroid = np.asarray(shape.centroid)
    long_ax  = np.asarray(shape.long_axis)
    short_ax = np.asarray(shape.short_axis)
    spans_m  = sorted(shape.extents_m, reverse=True)

    dim_str  = "  x  ".join(f"{d * 100:.1f} cm" for d in spans_m)
    dx_cm    = float(centroid[0]) * 100.0
    dy_cm    = float(centroid[1]) * 100.0

    jaw_xy   = np.array([long_ax[0], long_ax[1]])
    wrist_deg = (
        float(np.degrees(np.arctan2(jaw_xy[1], jaw_xy[0])))
        if np.linalg.norm(jaw_xy) > 0.1 else 0.0
    )

    jaw_span_cm = float(2.0 * np.sqrt(max((spans_m[0] / 5.0) ** 2, 0.0))) * 100.0
    jaw_open_cm = jaw_span_cm + 1.5

    _axis_desc = {0: "horizontal (left-right)", 1: "vertical (up-down)", 2: "axial (depth)"}
    jaw_desc   = _axis_desc[int(np.argmax(np.abs(long_ax)))]
    grasp_desc = _axis_desc[int(np.argmax(np.abs(short_ax)))]

    if spans_m[0] > 0.10:
        grip_type = "two-finger power grip (large object)"
    elif spans_m[0] < 0.04:
        grip_type = "precision pinch (small object)"
    else:
        grip_type = "standard parallel-jaw pinch"

    fill_pct = 100.0 * int(voxels.sum()) / voxels.size
    width    = 65
    sep      = "=" * width
    sub      = "-" * (width - 4)

    lines: List[str] = [
        sep,
        "  THEORETICAL GRASP PLAN  -  HEAPGrasp",
        f"  {n_views}-view reconstruction  |  camera distance {camera_distance * 100:.0f} cm",
        sep, "",
        "  OBJECT SUMMARY",
        f"  {'Estimated dimensions':<28}{dim_str}",
        f"  {'Jaw span axis':<28}{jaw_desc}",
        f"  {'Approach axis':<28}{grasp_desc}",
        f"  {'Recommended grip':<28}{grip_type}",
        f"  {'Reconstruction fill':<28}{fill_pct:.0f} %  ({int(voxels.sum()):,} occupied voxels)",
        "", f"  {sub}",
        "  1. LATERAL ALIGNMENT", f"  {sub}",
        "  Camera (mounted at gripper tip) sees the object at:",
    ]

    if abs(dx_cm) > 0.3:
        dirn = "right" if dx_cm > 0 else "left"
        axis = "right (+X)" if dx_cm > 0 else "left (-X)"
        lines += [f"    - {abs(dx_cm):.1f} cm to the {dirn} of optical centre",
                  f"      -> Translate TCP {abs(dx_cm):.1f} cm {axis}"]
    else:
        lines.append("    - Centred horizontally  (no X correction needed)")

    if abs(dy_cm) > 0.3:
        dirn = "above" if dy_cm > 0 else "below"
        axis = "up (+Y)" if dy_cm > 0 else "down (-Y)"
        lines += [f"    - {abs(dy_cm):.1f} cm {dirn} optical centre",
                  f"      -> Translate TCP {abs(dy_cm):.1f} cm {axis}"]
    else:
        lines.append("    - Centred vertically    (no Y correction needed)")

    lines += [
        "", f"  {sub}", "  2. WRIST ROTATION", f"  {sub}",
    ]
    if abs(wrist_deg) > 5.0:
        dirn = "counter-clockwise" if wrist_deg > 0 else "clockwise"
        lines += [
            f"  Jaw axis is {abs(wrist_deg):.0f} deg from horizontal in the image plane.",
            f"    -> Rotate wrist {abs(wrist_deg):.0f} deg  {dirn}",
        ]
    else:
        lines.append("  Jaw axis is approximately horizontal — no wrist rotation needed.")

    lines += [
        "", f"  {sub}", "  3. OPEN JAWS", f"  {sub}",
        f"  Object span along jaw axis:  {jaw_span_cm:.1f} cm",
        f"    -> Set jaw opening to  {jaw_open_cm:.1f} cm",
        f"       ({jaw_span_cm:.1f} cm object  +  1.5 cm clearance)",
        "", f"  {sub}", "  4. APPROACH", f"  {sub}",
        f"  Object is {camera_distance * 100:.0f} cm from the gripper tip.",
        f"    -> Advance {camera_distance * 100:.0f} cm along gripper Z-axis.",
        "    -> Speed: SLOW — transparent/specular surface, uncertain contact.",
        "", f"  {sub}", "  5. GRASP", f"  {sub}",
        "    -> Close jaws to contact resistance.",
        "    -> Monitor SlipDetectorCNN: stable → continue  |  slipping → re-seat.",
        "", f"  {sub}", "  6. RETRIEVE", f"  {sub}",
        "    -> Raise TCP 5 cm vertically to clear the surface.",
        "    -> Translate to deposit location at REDUCED speed.",
        "", sep, "  SAFETY NOTES",
        "  This plan is THEORETICAL — generated from camera data only.",
        "  A human operator must verify all clearances before executing.",
        f"  Reconstruction quality: {fill_pct:.0f}% voxel fill from {n_views} views.",
        sep,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Natural-language execution instructions for a single grasp
# ---------------------------------------------------------------------------

def format_grasp_instructions(grasp: "GraspCandidate", rank: int = 1) -> str:
    """
    Render step-by-step arm movement instructions for executing one grasp.

    All positions are in the world frame with the object centre at the origin.
    """
    d  = np.asarray(grasp.closing_dir,  dtype=float)
    a  = np.asarray(grasp.approach_dir, dtype=float)
    c  = np.asarray(grasp.center,       dtype=float)
    c1 = np.asarray(grasp.contact_1,    dtype=float)
    c2 = np.asarray(grasp.contact_2,    dtype=float)
    w  = grasp.width

    # ── Closing-direction description ─────────────────────────────────────
    dom_ax   = int(np.argmax(np.abs(d)))
    ax_label = ["X (left-right)", "Y (up-down)", "Z (front-back)"][dom_ax]
    vert_tilt_deg  = float(np.degrees(np.arcsin(np.clip(abs(d[1]), 0.0, 1.0))))
    horiz_angle_deg = float(np.degrees(np.arctan2(d[0], d[2])))

    # ── Approach-direction description ────────────────────────────────────
    dom_ap   = int(np.argmax(np.abs(a)))
    ap_sign  = int(np.sign(a[dom_ap]))
    _ap_desc = {
        (0,  1): "from the right  (+X)",
        (0, -1): "from the left   (−X)",
        (1,  1): "from above      (−Y, downward)",
        (1, -1): "from below      (+Y, upward)",
        (2,  1): "from the front  (+Z)",
        (2, -1): "from behind     (−Z)",
    }
    ap_desc = _ap_desc.get((dom_ap, ap_sign),
                           f"({a[0]:+.2f}, {a[1]:+.2f}, {a[2]:+.2f})")

    def _p(v: np.ndarray) -> str:
        return (f"X={v[0]*100:+.1f} cm, "
                f"Y={v[1]*100:+.1f} cm, "
                f"Z={v[2]*100:+.1f} cm")

    width_mm  = w * 1000.0
    open_mm   = width_mm + 15.0      # 15 mm safety clearance each side

    sep = "─" * 58
    lines = [
        sep,
        f"  GRASP #{rank} EXECUTION PLAN  "
        f"(score {grasp.score:.3f} | width {width_mm:.0f} mm)",
        sep, "",
        "  1. OPEN JAWS",
        f"     → Open to {open_mm:.0f} mm",
        f"       ({width_mm:.0f} mm object span + 15 mm clearance)",
        "",
        "  2. APPROACH",
        f"     → Move arm {ap_desc}",
        f"       Approach vector (world): "
        f"({a[0]:+.2f}, {a[1]:+.2f}, {a[2]:+.2f})",
        "",
        "  3. ALIGN TCP",
        f"     → TCP target from object centre:",
        f"       {_p(c)}",
        "",
        "  4. ORIENT WRIST  (align jaw axis)",
        f"     → Jaws close along dominant axis: {ax_label}",
        f"     → Closing vector (world): ({d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f})",
        f"     → Tilt from horizontal: {vert_tilt_deg:.0f}°",
        f"     → Horizontal rotation from front (Z): {horiz_angle_deg:.0f}°",
        "",
        "  5. CONTACT POINTS  (verify clearance)",
        f"     → Jaw A: {_p(c1)}",
        f"     → Jaw B: {_p(c2)}",
        "",
        "  6. CLOSE JAWS",
        "     → Close until contact resistance",
        f"     → Friction score: {grasp.normal_score:.2f}  "
        f"| Width score: {grasp.width_score:.2f}  "
        f"| Stability: {grasp.center_score:.2f}",
        "     → Monitor slip detector: stable → continue  |  slipping → re-seat",
        "",
        "  7. RETRIEVE",
        "     → Lift 5 cm vertically to clear object surface",
        "     → Translate to deposit location at reduced speed",
        "",
        sep,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default geometric scorer (fallback for LearnedGraspScorer)
# ---------------------------------------------------------------------------

def default_score(cand: GraspCandidate, points: np.ndarray) -> float:
    """
    Heuristic quality score in [0, 1] used as a fallback when no learned
    model is available. Rewards grasps close to the centroid with a
    comfortable width.
    """
    centroid = points.mean(0)
    dist = float(np.linalg.norm(cand.center - centroid))
    extent = float(np.linalg.norm(points.max(0) - points.min(0)))
    proximity = max(0.0, 1.0 - dist / (0.5 * extent + 1e-6))
    return float(0.65 * proximity + 0.35 * cand.score)


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


__all__ = [
    "GraspCandidate",
    "ObjectShape",
    "find_grasps",
    "grasp_from_voxels",
    "grasps_to_json",
    "format_grasp_plan",
    "default_score",
    "ScoreFn",
]
