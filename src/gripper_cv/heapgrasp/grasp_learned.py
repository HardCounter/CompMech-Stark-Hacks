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
  Supports both single-input models (our tiny stubs) and Dex-Net-style
  **dual-input** models (``image`` + ``pose`` gripper-depth scalar) with
  automatic NCHW / NHWC layout detection.
* **Hailo-8L (AI HAT+)** — optional; reuses
  :class:`gripper_cv.hailo.HailoRunner`. If the grasp network compiles to a
  `.hef`, it runs here with NPU acceleration.

Both backends are imported lazily, so the module can be imported on any
machine — tests exercise the geometric patch renderer without needing either.

Patch conventions
-----------------

The renderer supports two encodings so one module can drive both our custom
tiny stubs and stock Dex-Net 2.0 GQ-CNN weights:

* ``encoding="normalized"`` (default). A ``patch_size x patch_size`` float32
  map with values in ``[0, 1]``. 0 = far / empty, 1 = in contact.
* ``encoding="metric"`` (Dex-Net 2.0 convention). Raw depth in metres,
  centred on the grasp mid-plane. Empty cells are filled with the **median**
  of filled cells (standard Dex-Net preprocessing). A scalar
  ``gripper_depth_m`` is returned alongside the image so dual-input GQ-CNN
  graphs can be fed directly.

The local frame for both modes is::

    x_local = jaw_axis                  (width direction)
    z_local = approach direction        (gripper moves along this)
    y_local = z_local x x_local         (gripper height)

Presets
-------

Passing ``preset="gqcnn2"`` to :class:`LearnedGraspScorer` forces the metric
encoding and the canonical 0.1 m patch/depth span from the Dex-Net 2.0 paper.
When the loaded ONNX has ``preset=gqcnn2`` stored in its metadata (as produced
by ``scripts/export_gqcnn_onnx.py``) the preset is applied automatically under
``preset="auto"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import numpy as np

from .grasp import GraspCandidate, ScoreFn, default_score


Encoding = Literal["normalized", "metric"]
Preset = Literal["auto", "default", "gqcnn2"]


# Canonical Dex-Net 2.0 GQ-CNN hyper-parameters (paper § 4.2).
_GQCNN2_PATCH_SIZE = 32
_GQCNN2_PATCH_SPAN_M = 0.1
_GQCNN2_DEPTH_SPAN_M = 0.1
# Nominal wrist-camera height (metres). Dex-Net trained on RGB-D images
# captured from a wrist camera ~0.5 – 0.8 m above the workspace. Our visual
# hull does not have a real camera, so we fake a sensible scalar so the pose
# branch is never identically zero. Accuracy is secondary for the hackathon
# demo — Phase B fine-tunes this away.
_GQCNN2_NOMINAL_CAMERA_HEIGHT_M = 0.7


# ---------------------------------------------------------------------------
# Render output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatchRender:
    """
    Result of rendering a gripper-aligned depth patch.

    ``image`` semantics depend on the encoding the renderer was asked for:

    * ``encoding="normalized"`` — float32 in ``[0, 1]`` (0 far, 1 in contact).
    * ``encoding="metric"``      — float32 metres, centred on the grasp plane.

    ``gripper_depth_m`` is the Dex-Net 2.0 pose-branch scalar (distance from
    the nominal camera plane to the grasp centre, in metres). It is set only
    in metric mode; in normalised mode it defaults to 0.
    """

    image: np.ndarray
    gripper_depth_m: float = 0.0
    encoding: Encoding = "normalized"

    # Let callers keep treating the result as an ndarray for backward
    # compatibility with the pre-PatchRender API (e.g. ``patch.shape``,
    # ``patch.sum()``).
    def __array__(self, dtype=None) -> np.ndarray:
        return self.image.astype(dtype, copy=False) if dtype else self.image

    @property
    def shape(self) -> tuple[int, ...]:
        return self.image.shape

    def sum(self) -> float:
        return float(self.image.sum())

    def min(self) -> float:
        return float(self.image.min())

    def max(self) -> float:
        return float(self.image.max())


# ---------------------------------------------------------------------------
# Depth-patch renderer (pure NumPy, no ML dependencies)
# ---------------------------------------------------------------------------

def render_grasp_patch(
    points: np.ndarray,
    grasp: GraspCandidate,
    patch_size: int = 32,
    patch_span_m: float = 0.08,
    depth_span_m: float = 0.06,
    encoding: Encoding = "normalized",
    camera_height_m: float = _GQCNN2_NOMINAL_CAMERA_HEIGHT_M,
    return_render: bool = False,
) -> Union[np.ndarray, PatchRender]:
    """
    Render a gripper-aligned depth patch for GQ-CNN-style inference.

    Args:
        points:            (N, 3) point cloud in world frame (metres).
        grasp:             Candidate defining the local frame.
        patch_size:        H = W of the output patch (pixels).
        patch_span_m:      Physical width of the patch in the jaw plane (m).
        depth_span_m:      Maximum approach depth the patch captures (m);
                           points farther along -approach contribute 0.
        encoding:          "normalized" (default) or "metric" (Dex-Net 2.0).
        camera_height_m:   Nominal camera-to-workspace distance, used only in
                           metric mode to derive ``gripper_depth_m``.
        return_render:     If True, always return a :class:`PatchRender`.
                           Default keeps the historical ``np.ndarray`` return
                           for ``encoding="normalized"`` callers.

    Returns:
        * ``encoding="normalized"`` and ``return_render=False`` (default):
          ``np.ndarray`` of shape ``(patch_size, patch_size)`` float32 in
          ``[0, 1]`` (backward compatible).
        * Otherwise: :class:`PatchRender`.
    """
    image, gripper_depth_m = _render_core(
        points, grasp,
        patch_size=patch_size,
        patch_span_m=patch_span_m,
        depth_span_m=depth_span_m,
        encoding=encoding,
        camera_height_m=camera_height_m,
    )
    if return_render or encoding != "normalized":
        return PatchRender(
            image=image,
            gripper_depth_m=gripper_depth_m,
            encoding=encoding,
        )
    return image


def _render_core(
    points: np.ndarray,
    grasp: GraspCandidate,
    *,
    patch_size: int,
    patch_span_m: float,
    depth_span_m: float,
    encoding: Encoding,
    camera_height_m: float,
) -> tuple[np.ndarray, float]:
    if len(points) == 0:
        return np.zeros((patch_size, patch_size), dtype=np.float32), 0.0

    pos = np.asarray(grasp.position, dtype=np.float64)
    x_local = _unit(np.asarray(grasp.jaw_axis, dtype=np.float64))
    z_local = _unit(np.asarray(grasp.approach, dtype=np.float64))
    y_local = _unit(np.cross(z_local, x_local))
    # Re-orthogonalise in case approach and jaw_axis are not perfectly orthogonal.
    x_local = _unit(np.cross(y_local, z_local))

    rel = points - pos                                # (N, 3)
    u = rel @ x_local                                 # width
    v = rel @ y_local                                 # height
    depth = -(rel @ z_local)                          # +ve in front of gripper

    half = patch_span_m / 2.0
    in_plane = (
        (np.abs(u) < half) & (np.abs(v) < half)
        & (depth >= 0.0) & (depth < depth_span_m)
    )

    # Dex-Net scalar: camera-plane → grasp-centre distance along +Z world.
    # Falls back to the nominal height when the grasp is near the workspace.
    gripper_depth_m = float(max(0.0, camera_height_m - pos[2]))

    if not in_plane.any():
        return np.zeros((patch_size, patch_size), dtype=np.float32), gripper_depth_m

    cell = patch_span_m / patch_size
    ui = np.clip(((u[in_plane] + half) / cell).astype(np.int32), 0, patch_size - 1)
    vi = np.clip(((v[in_plane] + half) / cell).astype(np.int32), 0, patch_size - 1)
    d = depth[in_plane]

    # Keep the *closest* point per cell (min depth = in front).
    patch = np.full((patch_size, patch_size), np.inf, dtype=np.float32)
    for uu, vv, dd in zip(ui, vi, d):
        if dd < patch[vv, uu]:
            patch[vv, uu] = dd

    filled = np.isfinite(patch)
    if encoding == "normalized":
        out = np.zeros((patch_size, patch_size), dtype=np.float32)
        out[filled] = np.clip(1.0 - patch[filled] / depth_span_m, 0.0, 1.0)
        return out, gripper_depth_m

    # Metric / Dex-Net convention: raw depth in metres, centred on the grasp
    # mid-plane, with the median of filled cells filling the empties.
    centre = depth_span_m / 2.0
    metric = patch.copy()
    if filled.any():
        median = float(np.median(metric[filled]))
    else:
        median = centre
    metric[~filled] = median
    metric = (metric - centre).astype(np.float32)
    return metric, gripper_depth_m


# ---------------------------------------------------------------------------
# Backend wrappers (lazy imports)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Backend:
    predict: Callable[[np.ndarray, float], float]
    close: Callable[[], None]
    name: str
    # Layout hints consumed by the scorer so the image tensor arrives in the
    # shape the graph actually expects.
    image_layout: Literal["NCHW", "NHWC"] = "NCHW"
    wants_pose: bool = False
    onnx_metadata: dict = None  # type: ignore[assignment]


def _make_onnx_backend(
    onnx_path: str,
    input_name: Optional[str],
) -> _Backend:
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
    inputs = session.get_inputs()

    # Gather metadata_props if present (emitted by scripts/export_gqcnn_onnx.py).
    meta = {}
    try:
        meta = dict(session.get_modelmeta().custom_metadata_map or {})
    except Exception:  # noqa: BLE001
        meta = {}

    image_layout: Literal["NCHW", "NHWC"] = "NCHW"
    wants_pose = False

    if len(inputs) == 1:
        image_name = input_name or inputs[0].name
        image_layout = _infer_image_layout(inputs[0].shape, meta)
        pose_name: Optional[str] = None
    elif len(inputs) == 2:
        wants_pose = True
        image_in, pose_in = _identify_dual_inputs(inputs, meta)
        image_name = input_name or image_in.name
        pose_name = pose_in.name
        image_layout = _infer_image_layout(image_in.shape, meta)
    else:
        raise ValueError(
            f"Unsupported ONNX input count {len(inputs)}; expected 1 or 2 "
            "(image [+ pose])."
        )

    def _predict(image_nchw: np.ndarray, gripper_depth_m: float) -> float:
        tensor = image_nchw.astype(np.float32)
        if image_layout == "NHWC":
            # (N, C, H, W) → (N, H, W, C)
            tensor = np.transpose(tensor, (0, 2, 3, 1))
        feeds = {image_name: tensor}
        if pose_name is not None:
            feeds[pose_name] = np.array(
                [[gripper_depth_m]], dtype=np.float32
            )
        out = session.run(None, feeds)
        return _softmax_last_positive(out[0])

    return _Backend(
        predict=_predict,
        close=lambda: None,
        name="onnx",
        image_layout=image_layout,
        wants_pose=wants_pose,
        onnx_metadata=meta,
    )


def _identify_dual_inputs(inputs, meta):
    """Map two ONNX inputs to (image, pose) by metadata first, then shape."""
    img_name_hint = meta.get("image_input")
    pose_name_hint = meta.get("pose_input")
    if img_name_hint and pose_name_hint:
        by_name = {i.name: i for i in inputs}
        if img_name_hint in by_name and pose_name_hint in by_name:
            return by_name[img_name_hint], by_name[pose_name_hint]

    # Fallback: 4-D tensor is the image; 2-D (or 1-D) is the pose.
    ranks = [len(i.shape) for i in inputs]
    if 4 in ranks and any(r <= 2 for r in ranks):
        image_in = inputs[ranks.index(4)]
        pose_in = inputs[1 - ranks.index(4)]
        return image_in, pose_in
    raise ValueError(
        "Could not identify image vs pose inputs. Expect one 4-D tensor "
        "(image) and one ≤2-D tensor (pose), or explicit ONNX metadata."
    )


def _infer_image_layout(shape, meta) -> Literal["NCHW", "NHWC"]:
    declared = (meta or {}).get("image_layout")
    if declared in {"NCHW", "NHWC"}:
        return declared  # type: ignore[return-value]
    # Shape may be symbolic ("batch") — inspect the last dim for a channel hint.
    dims = [d if isinstance(d, int) else -1 for d in shape]
    if len(dims) == 4 and dims[-1] in (1, 3) and dims[1] not in (1, 3):
        return "NHWC"
    return "NCHW"


def _make_hailo_backend(hef_path: str) -> _Backend:
    from gripper_cv.hailo import HailoRunner

    runner = HailoRunner(hef_path)

    def _predict(image_nchw: np.ndarray, gripper_depth_m: float) -> float:
        # Hailo compiled graphs we produce are single-input; the pose branch is
        # either baked in or dropped at compile time.
        out = runner.run_single(image_nchw.astype(np.float32))
        return _softmax_last_positive(out)

    return _Backend(
        predict=_predict, close=runner.close, name="hailo",
    )


# ---------------------------------------------------------------------------
# High-level scorer
# ---------------------------------------------------------------------------

class LearnedGraspScorer:
    """
    GQ-CNN-style grasp quality predictor.

    Typical use::

        # Custom single-input ONNX stub
        scorer = LearnedGraspScorer(onnx_path="gqcnn_stub.onnx")

        # Stock Dex-Net 2.0 GQ-CNN 2.0 (dual-input, metric depth)
        scorer = LearnedGraspScorer(
            onnx_path="models/gqcnn_2.0.onnx", preset="gqcnn2",
        )

        cands = sample_grasps(points, score_fn=scorer.score_fn)

    When no model is supplied (or the backend import fails) the scorer falls
    back to :func:`gripper_cv.heapgrasp.grasp.default_score`, so callers can
    always rely on a non-trivial ranking.

    Args:
        onnx_path:         Path to a compiled ONNX grasp quality model.
        hef_path:          Path to a compiled Hailo .hef model (NPU). Mutually
                           exclusive with ``onnx_path`` — if both are provided
                           ``hef_path`` wins.
        patch_size:        Height / width of the depth patch in pixels.
        patch_span_m:      Physical span the patch covers (jaw plane).
        depth_span_m:      Maximum approach depth encoded into the patch.
        encoding:          Patch encoding ("normalized" or "metric"). Ignored
                           when a non-default preset sets its own value.
        input_name:        Optional ONNX input tensor name for the image
                           (defaults to the session's first input or the name
                           declared in ONNX metadata).
        preset:            "auto" (default) reads the ONNX metadata and picks
                           the right convention. "default" keeps the normalised
                           [0, 1] patches. "gqcnn2" forces Dex-Net 2.0
                           conventions (32x32, 0.1 m span, metric encoding).
    """

    def __init__(
        self,
        onnx_path: Optional[str | Path] = None,
        hef_path: Optional[str | Path] = None,
        patch_size: int = 32,
        patch_span_m: float = 0.08,
        depth_span_m: float = 0.06,
        encoding: Encoding = "normalized",
        input_name: Optional[str] = None,
        preset: Preset = "auto",
    ) -> None:
        self.patch_size = patch_size
        self.patch_span_m = patch_span_m
        self.depth_span_m = depth_span_m
        self.encoding: Encoding = encoding
        self.preset: Preset = preset

        self._backend: Optional[_Backend] = None
        if hef_path is not None:
            self._backend = _make_hailo_backend(str(hef_path))
        elif onnx_path is not None:
            self._backend = _make_onnx_backend(str(onnx_path), input_name)

        # Apply preset last so ONNX metadata can drive "auto" detection.
        resolved = self._resolve_preset()
        if resolved == "gqcnn2":
            self._apply_gqcnn2_preset()

    # ------------------------------------------------------------------
    # Preset handling
    # ------------------------------------------------------------------

    def _resolve_preset(self) -> Preset:
        if self.preset != "auto":
            return self.preset
        meta = (self._backend.onnx_metadata
                if self._backend and self._backend.onnx_metadata else {})
        declared = meta.get("preset")
        if declared in {"gqcnn2", "default"}:
            return declared  # type: ignore[return-value]
        # Heuristic: any ONNX model that wants two inputs is almost certainly
        # Dex-Net-flavoured.
        if self._backend and self._backend.wants_pose:
            return "gqcnn2"
        return "default"

    def _apply_gqcnn2_preset(self) -> None:
        self.patch_size = _GQCNN2_PATCH_SIZE
        self.patch_span_m = _GQCNN2_PATCH_SPAN_M
        self.depth_span_m = _GQCNN2_DEPTH_SPAN_M
        self.encoding = "metric"
        self.preset = "gqcnn2"

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

    def render(self, points: np.ndarray, grasp: GraspCandidate) -> PatchRender:
        return render_grasp_patch(
            points, grasp,
            patch_size=self.patch_size,
            patch_span_m=self.patch_span_m,
            depth_span_m=self.depth_span_m,
            encoding=self.encoding,
            return_render=True,
        )  # type: ignore[return-value]

    def predict(self, points: np.ndarray, grasp: GraspCandidate) -> float:
        if self._backend is None:
            return default_score(grasp, points)

        render = self.render(points, grasp)
        # NCHW with a single grey channel — the canonical GQ-CNN layout; the
        # backend will transpose to NHWC internally if needed.
        image_nchw = render.image[np.newaxis, np.newaxis, :, :]
        return float(self._backend.predict(image_nchw, render.gripper_depth_m))

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
    "PatchRender",
    "render_grasp_patch",
]
