"""
GQ-CNN-style learned grasp scorer.

This module plugs a small 2.5-D CNN into
:func:`gripper_cv.heapgrasp.grasp.sample_grasps` as the ``score_fn`` hook.
The network consumes a gripper-aligned depth patch rendered from the visual
hull point cloud and predicts a grasp-quality probability in [0, 1], exactly
like Dex-Net 2.0 / GQ-CNN.

Runtime backends
----------------
* **ONNX Runtime** — CPU-only default; works on the Pi 5 without the AI HAT.
* **Hailo-8L (AI HAT+)** — optional; reuses
  :class:`gripper_cv.hailo.HailoRunner`. If the grasp network compiles to a
  `.hef`, it runs here with NPU acceleration.

Both backends are imported lazily, so the module can be imported on any
machine — tests exercise the geometric patch renderer without needing either.

Why not GraspNet-1Billion?
--------------------------
Full point-cloud grasp networks (GraspNet-1Billion etc.) are large enough
that running them on the Pi 5 without heavy distillation is not realistic
for the hackathon demo. The 2.5-D patch approach is the standard edge
compromise: the network is small, input is a 32x32 depth image, and the
output is a single scalar.

Patch convention (input to the network)
---------------------------------------
A gripper-aligned depth patch is a 1 x H x W float32 tensor with values in
[0, 1] representing normalised depth in front of the gripper. The local
frame is::

    x_local = jaw_axis                  (width direction)
    z_local = approach direction        (gripper moves along this)
    y_local = z_local x x_local         (gripper height)

0 = far (no geometry), 1 = in contact. Empty cells are encoded as 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .grasp import GraspCandidate, ScoreFn, default_score


# ---------------------------------------------------------------------------
# Depth-patch renderer (pure NumPy, no ML dependencies)
# ---------------------------------------------------------------------------

def render_grasp_patch(
    points: np.ndarray,
    grasp: GraspCandidate,
    patch_size: int = 32,
    patch_span_m: float = 0.08,
    depth_span_m: float = 0.06,
) -> np.ndarray:
    """
    Render a gripper-aligned depth patch for GQ-CNN-style inference.

    Args:
        points:        (N, 3) point cloud in world frame (metres).
        grasp:         Candidate defining the local frame.
        patch_size:    H = W of the output patch (pixels).
        patch_span_m:  Physical width of the patch in the jaw plane (metres).
        depth_span_m:  Maximum approach depth the patch captures (metres);
                       points farther than this along -approach contribute 0.

    Returns:
        (patch_size, patch_size) float32 depth map in [0, 1].
    """
    if len(points) == 0:
        return np.zeros((patch_size, patch_size), dtype=np.float32)

    pos = np.asarray(grasp.position, dtype=np.float64)
    x_local = _unit(np.asarray(grasp.jaw_axis, dtype=np.float64))
    z_local = _unit(np.asarray(grasp.approach, dtype=np.float64))
    y_local = _unit(np.cross(z_local, x_local))
    # Re-orthogonalise in case approach and jaw_axis are not perfectly orthogonal
    x_local = _unit(np.cross(y_local, z_local))

    rel = points - pos                                # (N, 3)
    u = rel @ x_local                                 # width direction
    v = rel @ y_local                                 # height direction
    depth = -(rel @ z_local)                          # +ve in front of gripper

    half = patch_span_m / 2.0
    in_plane = (
        (np.abs(u) < half) & (np.abs(v) < half)
        & (depth >= 0.0) & (depth < depth_span_m)
    )
    if not in_plane.any():
        return np.zeros((patch_size, patch_size), dtype=np.float32)

    cell = patch_span_m / patch_size
    ui = np.clip(
        ((u[in_plane] + half) / cell).astype(np.int32),
        0, patch_size - 1,
    )
    vi = np.clip(
        ((v[in_plane] + half) / cell).astype(np.int32),
        0, patch_size - 1,
    )
    d = depth[in_plane]

    # Keep the *closest* point per cell (min depth = in front).
    patch = np.full((patch_size, patch_size), np.inf, dtype=np.float32)
    for uu, vv, dd in zip(ui, vi, d):
        if dd < patch[vv, uu]:
            patch[vv, uu] = dd

    filled = np.isfinite(patch)
    out = np.zeros((patch_size, patch_size), dtype=np.float32)
    # Closer depth → brighter; empty cells stay 0.
    out[filled] = np.clip(1.0 - patch[filled] / depth_span_m, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Backend wrappers (lazy imports)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Backend:
    predict: Callable[[np.ndarray], float]
    close: Callable[[], None]
    name: str


def _make_onnx_backend(onnx_path: str, input_name: Optional[str]) -> _Backend:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is not installed. Install with:\n"
            "  pip install onnxruntime"
        ) from exc

    session = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    name = input_name or session.get_inputs()[0].name

    def _predict(patch_nchw: np.ndarray) -> float:
        out = session.run(None, {name: patch_nchw.astype(np.float32)})
        return _softmax_last_positive(out[0])

    return _Backend(predict=_predict, close=lambda: None, name="onnx")


def _make_hailo_backend(hef_path: str) -> _Backend:
    from gripper_cv.hailo import HailoRunner

    runner = HailoRunner(hef_path)

    def _predict(patch_nchw: np.ndarray) -> float:
        out = runner.run_single(patch_nchw.astype(np.float32))
        return _softmax_last_positive(out)

    return _Backend(predict=_predict, close=runner.close, name="hailo")


# ---------------------------------------------------------------------------
# High-level scorer
# ---------------------------------------------------------------------------

class LearnedGraspScorer:
    """
    GQ-CNN-style grasp quality predictor.

    Typical use::

        scorer = LearnedGraspScorer(onnx_path="gqcnn.onnx")
        cands = sample_grasps(points, score_fn=scorer.score_fn)

    When no model is supplied (or the backend import fails) the scorer falls
    back to :func:`gripper_cv.heapgrasp.grasp.default_score`, so callers can
    always rely on a non-trivial ranking.

    Args:
        onnx_path:     Path to a compiled ONNX grasp quality model.
        hef_path:      Path to a compiled Hailo .hef model (NPU).  Mutually
                       exclusive with ``onnx_path`` — if both are provided
                       ``hef_path`` wins.
        patch_size:    Height / width of the depth patch in pixels.
        patch_span_m:  Physical span the patch covers (in the jaw plane).
        depth_span_m:  Maximum approach depth encoded into the patch.
        input_name:    Optional ONNX input tensor name (defaults to the
                       session's first input).
    """

    def __init__(
        self,
        onnx_path: Optional[str | Path] = None,
        hef_path: Optional[str | Path] = None,
        patch_size: int = 32,
        patch_span_m: float = 0.08,
        depth_span_m: float = 0.06,
        input_name: Optional[str] = None,
    ) -> None:
        self.patch_size = patch_size
        self.patch_span_m = patch_span_m
        self.depth_span_m = depth_span_m

        self._backend: Optional[_Backend] = None
        if hef_path is not None:
            self._backend = _make_hailo_backend(str(hef_path))
        elif onnx_path is not None:
            self._backend = _make_onnx_backend(str(onnx_path), input_name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        """Name of the active backend ('onnx', 'hailo', or 'fallback')."""
        return self._backend.name if self._backend else "fallback"

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    def __enter__(self) -> "LearnedGraspScorer":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Patch + inference
    # ------------------------------------------------------------------

    def render(self, points: np.ndarray, grasp: GraspCandidate) -> np.ndarray:
        return render_grasp_patch(
            points, grasp,
            patch_size=self.patch_size,
            patch_span_m=self.patch_span_m,
            depth_span_m=self.depth_span_m,
        )

    def predict(self, points: np.ndarray, grasp: GraspCandidate) -> float:
        if self._backend is None:
            return default_score(grasp, points)

        patch = self.render(points, grasp)
        # NCHW with a single grey channel — the GQ-CNN convention.
        patch_nchw = patch[np.newaxis, np.newaxis, :, :]
        return float(self._backend.predict(patch_nchw))

    @property
    def score_fn(self) -> ScoreFn:
        """Return a callable for sample_grasps(score_fn=...)."""

        def _score(cand: GraspCandidate, points: np.ndarray) -> float:
            return self.predict(points, cand)

        return _score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-9 else v / n


def _softmax_last_positive(arr: np.ndarray) -> float:
    """
    Normalise a classifier output into a [0, 1] probability.

    Handles three common GQ-CNN-style head layouts:

    * scalar / (1,)                      — sigmoid pass-through
    * (1, 1)                             — sigmoid pass-through
    * (1, 2) logits (negative, positive) — softmax, return positive probability
    """
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    if a.size == 1:
        x = float(a[0])
        return float(1.0 / (1.0 + np.exp(-x))) if x < 0 or x > 1.0 else x
    if a.size == 2:
        e = np.exp(a - a.max())
        return float(e[1] / e.sum())
    return float(a[-1])


__all__ = [
    "LearnedGraspScorer",
    "render_grasp_patch",
]
