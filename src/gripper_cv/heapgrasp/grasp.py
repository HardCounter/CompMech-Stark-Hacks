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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .reconstruct import voxels_to_pointcloud


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


def _extract_surface(
    voxels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (indices, normals) for all surface voxels.

    A surface voxel is occupied with at least one empty 6-connected neighbour.
    The outward normal is the normalised sum of directions toward empty neighbours.

    Returns
    -------
    indices : (N, 3) int   — (i, j, k) grid indices of surface voxels
    normals : (N, 3) float — unit outward normals
    """
    occ = voxels.astype(bool)
    is_surface = np.zeros(occ.shape, dtype=bool)
    normals = np.zeros(occ.shape + (3,), dtype=np.float32)

    for ax in range(3):
        for sign in (1, -1):
            # Roll neighbour into place; blank the wrapped edge.
            nbr = np.roll(occ, sign, axis=ax)
            edge = [slice(None)] * 3
            edge[ax] = 0 if sign > 0 else -1
            nbr[tuple(edge)] = False

            cond = occ & ~nbr                    # occupied, empty neighbour
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

    # Surface voxels → world positions and outward normals
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
    ], axis=1)                              # (N_surf, 3)

    # Centroid from full point cloud
    centroid = pts_all.mean(0)

    # Candidate closing directions: PCA axes + hemisphere sample
    cov = np.cov((pts_all - centroid).T)
    _, eigvecs = np.linalg.eigh(cov)       # ascending eigenvalues
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

        # Extreme surface points along closing direction
        projs = surf_pts @ d
        i1, i2 = int(np.argmin(projs)), int(np.argmax(projs))
        width = float(projs[i2] - projs[i1])

        if width < 0.003 or width > max_jaw_width:
            continue

        # De-duplicate: skip nearly-parallel directions
        if any(abs(float(np.dot(d, s))) > 0.95 for s in seen_dirs):
            continue

        c1 = surf_pts[i1]
        c2 = surf_pts[i2]
        n1 = surf_normals[i1]
        n2 = surf_normals[i2]

        # Friction-cone proxy: normals should oppose the jaw direction
        # n1 faces away from object at c1, should point in -d (toward jaw A)
        # n2 faces away from object at c2, should point in +d (toward jaw B)
        f1 = float(np.dot(n1, -d))
        f2 = float(np.dot(n2,  d))
        normal_score = float(np.clip((f1 + f2) / 2.0, 0.0, 1.0))

        # Width score: Gaussian around ideal opening
        width_score = float(
            np.exp(-((width - ideal_width) / (ideal_width * 0.5)) ** 2)
        )

        # Center score: TCP close to object centroid
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
