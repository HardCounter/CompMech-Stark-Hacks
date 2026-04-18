"""
Next-Best-View planner for HEAPGrasp active perception.

Plans the next camera viewpoint using an optimised cost-function approach:

    Score(θ) = IG(θ) / (1 + λ · min_angular_distance(θ, visited))

    IG(θ) = Σ_v  occupied(v) · in_frame(v, θ) · (1 / (1 + times_seen(v)))

where times_seen(v) counts how many past views included voxel v in their image
frustum.  Voxels observed by fewer views are more uncertain and contribute more
information gain.  The trajectory term penalises long camera moves.

All computation is pure NumPy — no torch dependency.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _angular_diff(a: float, b: float) -> float:
    """Minimum angular distance between two angles in degrees (wraps at 360)."""
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


class NextBestViewPlanner:
    """
    Plans the optimal next turntable angle using the IG cost function.

    Usage pattern (one update call per captured view):

        planner = NextBestViewPlanner(K, volume_size=64)
        planner.update(0.0, initial_voxels)
        next_angle = planner.next_best_view()   # → e.g. 180.0
        planner.update(next_angle, updated_voxels)
        ...

    Args:
        camera_matrix:   (3, 3) intrinsic matrix K — same as passed to
                         shape_from_silhouette()
        volume_size:     voxel grid side length V (grid = V³ voxels)
        object_diameter: reconstruction volume side in metres
        camera_distance: camera-to-object-centre distance in metres
        lam:             λ — trajectory penalty weight (higher → prefer nearby views)
        n_candidates:    number of evenly-spaced candidate angles evaluated per call
        image_width:     camera frame width in pixels (default 640)
        image_height:    camera frame height in pixels (default 480)
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        volume_size: int = 64,
        object_diameter: float = 0.15,
        camera_distance: float = 0.40,
        lam: float = 0.5,
        n_candidates: int = 36,
        image_width: int = 640,
        image_height: int = 480,
    ) -> None:
        self.K = camera_matrix.astype(np.float64)
        self.V = volume_size
        self.object_diameter = object_diameter
        self.camera_distance = camera_distance
        self.lam = lam
        self.n_candidates = n_candidates
        self._W = image_width
        self._H = image_height

        # Voxel world-space coordinates: (3, V³) — built once and reused
        half = object_diameter / 2.0
        lin = np.linspace(-half, half, volume_size)
        xi, yi, zi = np.meshgrid(lin, lin, lin, indexing="ij")
        self._pts = np.stack([xi.ravel(), yi.ravel(), zi.ravel()])  # (3, V³)

        # Mutable state — updated by update()
        self._voxels: np.ndarray = np.ones((volume_size,) * 3, dtype=bool)
        self._times_seen: np.ndarray = np.zeros((volume_size,) * 3, dtype=np.int32)
        self._visited: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, angle_deg: float, voxels: np.ndarray) -> None:
        """
        Record a completed view at angle_deg with the resulting occupancy grid.

        Increments times_seen for all voxels in-frame at angle_deg and stores
        the current occupancy as the reference for the next IG computation.
        """
        in_frame = self._in_frame_mask(angle_deg).reshape(self.V, self.V, self.V)
        self._times_seen += in_frame.astype(np.int32)
        self._voxels = voxels.astype(bool)
        self._visited.append(angle_deg)

    def score(self, theta_deg: float) -> float:
        """Evaluate Score(θ) for a single candidate angle."""
        ig = self._information_gain(theta_deg)
        if not self._visited:
            return ig
        min_dist = min(_angular_diff(theta_deg, v) for v in self._visited)
        return ig / (1.0 + self.lam * min_dist)

    def scores_all(self) -> Dict[float, float]:
        """Return {angle: score} for all n_candidates evenly-spaced angles."""
        step = 360.0 / self.n_candidates
        return {i * step: self.score(i * step) for i in range(self.n_candidates)}

    def next_best_view(self) -> float:
        """
        Return the candidate angle with the highest Score(θ).

        Angles within 2° of an already-visited view are skipped; if all
        candidates are exhausted (full rotation done), the full set is used.
        """
        step = 360.0 / self.n_candidates
        candidates = [i * step for i in range(self.n_candidates)]
        unvisited = [
            c for c in candidates
            if not any(_angular_diff(c, v) < 2.0 for v in self._visited)
        ]
        pool = unvisited if unvisited else candidates
        return max(pool, key=self.score)

    def suggest_view_schedule(
        self,
        n_total: int = 12,
        n_initial: int = 4,
    ) -> List[float]:
        """
        Suggest an ordered list of view angles for a full capture session.

        The first n_initial angles are evenly spaced (no prior reconstruction).
        Remaining angles are chosen greedily using a simulated sphere visual hull
        computed from the initial views.

        Returns a fresh list; does not modify planner state.
        """
        from gripper_cv.heapgrasp.reconstruct import shape_from_silhouette

        angles: List[float] = [i * 360.0 / n_initial for i in range(n_initial)]
        if n_total <= n_initial:
            return angles

        # Simulate circular (sphere) silhouettes for the initial views
        H, W = self._H, self._W
        masks = []
        for _ in angles:
            m = np.zeros((H, W), dtype=bool)
            cy, cx = H // 2, W // 2
            r = min(H, W) // 3
            yg, xg = np.ogrid[:H, :W]
            m[(yg - cy) ** 2 + (xg - cx) ** 2 <= r ** 2] = True
            masks.append(m)

        voxels = shape_from_silhouette(
            masks, angles, self.K,
            self.V, self.object_diameter, self.camera_distance,
        )

        # Use a temporary planner so we don't mutate self
        tmp = NextBestViewPlanner(
            self.K, self.V, self.object_diameter, self.camera_distance,
            self.lam, self.n_candidates, self._W, self._H,
        )
        for ang in angles:
            tmp.update(ang, voxels)

        for _ in range(n_total - n_initial):
            nbv = tmp.next_best_view()
            angles.append(nbv)
            tmp.update(nbv, voxels)

        return angles

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _in_frame_mask(self, theta_deg: float) -> np.ndarray:
        """
        Return (V³,) bool — True for each voxel whose projection falls
        inside the image frame when the camera is at angle theta_deg.

        Uses the same turntable camera model as reconstruct.shape_from_silhouette:
        camera orbits by −theta_deg around Y; object stays at the origin.
        """
        theta = np.radians(-theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        Ry = np.array(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64
        )

        pts_cam = Ry @ self._pts                   # (3, V³)
        pts_cam = pts_cam.copy()                   # avoid in-place aliasing
        pts_cam[2] += self.camera_distance

        depth = pts_cam[2]
        in_front = depth > 1e-3

        proj = self.K @ pts_cam                    # (3, V³)
        u = proj[0] / (depth + 1e-9)
        v = proj[1] / (depth + 1e-9)
        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)

        return in_front & (ui >= 0) & (ui < self._W) & (vi >= 0) & (vi < self._H)

    def _information_gain(self, theta_deg: float) -> float:
        """
        Compute IG(θ) via a vectorised dot product (sub-ms for V=64).

        IG = (occupied & in_frame)^T · (1 / (1 + times_seen))
        """
        if not self._voxels.any():
            return 0.0
        in_frame = self._in_frame_mask(theta_deg)           # (V³,) bool
        occ = self._voxels.ravel()                           # (V³,) bool
        inv_seen = 1.0 / (1.0 + self._times_seen.ravel().astype(np.float32))
        return float((occ & in_frame).astype(np.float32) @ inv_seen)
