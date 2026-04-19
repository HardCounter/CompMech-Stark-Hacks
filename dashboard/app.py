"""
HEAPGrasp — Interactive Demo Dashboard  v3
Run with:  streamlit run dashboard/app.py

The Pi camera (if connected) feeds directly into the capture wizard via
st.camera_input().  Captured frames are stored in session_state and flow
automatically into the 3D scan and grasp-planning pages.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

# Detect real Pi camera BEFORE adding any mock.
# setdefault would block the real module if it's already importable.
_PI_CAMERA_AVAILABLE = False
try:
    import picamera2 as _picamera2_probe          # noqa: F401
    _PI_CAMERA_AVAILABLE = True
except (ImportError, RuntimeError, Exception):
    # Not on Pi, or picamera2 not installed — mock it so gripper_cv imports work
    sys.modules["picamera2"] = MagicMock()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gripper_cv.hailo import is_hailo_available
from gripper_cv.heapgrasp.capture import CaptureSession
from gripper_cv.heapgrasp.grasp import GraspCandidate, find_grasps, format_grasp_plan
from gripper_cv.heapgrasp.reconstruct import (
    default_camera_matrix,
    estimate_grid_center,
    reproject_hull,
    shape_from_silhouette,
    voxels_to_pointcloud,
)
from gripper_cv.heapgrasp.segment import _apply_clahe, extract_silhouettes, remove_shadows
from gripper_cv.planner import NextBestViewPlanner
from gripper_cv.sim2real.domain_rand import DomainRandomTransform

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HEAPGrasp · AI Gripper",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetric"] {
    background:#1e1e2e; border:1px solid #313244;
    border-radius:8px; padding:10px 16px;
}
.pipeline-step {
    background:#181825; border:1px solid #313244;
    border-radius:10px; padding:16px; text-align:center;
}
.step-done   { border-color:#a6e3a1 !important; }
.step-active { border-color:#89b4fa !important; }
.hailo-pill {
    display:inline-block; border-radius:20px;
    padding:4px 14px; font-size:.82em; font-weight:700; letter-spacing:.03em;
}
.hailo-ok  { background:#a6e3a1; color:#1e1e2e; }
.hailo-off { background:#f38ba8; color:#1e1e2e; }
.grasp-plan {
    background:#1e1e2e; border:1px solid #45475a;
    border-radius:8px; padding:20px;
    font-family:monospace; white-space:pre; font-size:.82em;
    line-height:1.55; overflow-x:auto;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hailo probe
# ---------------------------------------------------------------------------

_HAILO_OK = is_hailo_available()
_HAILO_VERSION = ""
if _HAILO_OK:
    try:
        import hailo_platform as _hp
        _HAILO_VERSION = getattr(_hp, "__version__", "")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Persistent capture session state
# ---------------------------------------------------------------------------

if "cap_bg" not in st.session_state:
    st.session_state.cap_bg: Optional[np.ndarray] = None
    st.session_state.cap_frames: List[np.ndarray] = []
    st.session_state.cap_angles: List[float] = []
    st.session_state.cap_n_views: int = 8
    # Polar angle θ from zenith at which all views were captured.
    # 90° = horizontal side view (turntable), 36° ≈ π/5 = HEAPGrasp hand-eye.
    st.session_state.cap_theta_deg: float = 90.0

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🦾 HEAPGrasp")
    st.caption("Ford StarkHacks · Raspberry Pi 5")
    st.markdown("---")

    _pill_cls  = "hailo-ok" if _HAILO_OK else "hailo-off"
    _pill_text = (f"✅ Hailo-8L &nbsp;{_HAILO_VERSION}" if _HAILO_OK
                  else "⚠️ Hailo: not detected")
    st.markdown(f"<span class='hailo-pill {_pill_cls}'>{_pill_text}</span>",
                unsafe_allow_html=True)

    st.markdown("---")

    n_done = len(st.session_state.cap_frames)
    n_tot  = st.session_state.cap_n_views
    has_bg = st.session_state.cap_bg is not None
    if has_bg or n_done > 0:
        st.markdown("**Capture session**")
        st.progress(n_done / n_tot,
                    text=f"{'BG ✓ · ' if has_bg else ''}{n_done}/{n_tot} views")
        if n_done == n_tot:
            st.success("Session complete — go to 3D Object Scan")
    else:
        st.caption("No active capture session.")

    st.markdown("---")
    PAGES = [
        "🏠 Overview",
        "🎬 Live Capture",
        "🔍 Image Classifier",
        "✂️ Segmentation",
        "📦 3D Object Scan",
        "⚡ Hailo NPU",
    ]
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

# ---------------------------------------------------------------------------
# Pi camera resource (opened once, kept alive across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def _open_pi_camera():
    """
    Open PiCameraStream and cache it for the lifetime of the Streamlit process.
    Returns the stream object, or None if the Pi camera is unavailable.
    """
    if not _PI_CAMERA_AVAILABLE:
        return None
    try:
        from gripper_cv.camera import CameraConfig, PiCameraStream
        cam = PiCameraStream(CameraConfig(width=640, height=480, fps=30))
        cam.start()
        return cam
    except Exception as exc:
        # Camera physically absent or initialisation failed
        st.warning(f"Pi camera init failed: {exc}")
        return None


def _pi_capture_ui(
    label: str,
    key_prefix: str,
    background: Optional[np.ndarray] = None,
    contrast_enhance: bool = False,
    clahe_clip: float = 2.0,
    shadow_removal: bool = True,
    shadow_darkness: float = 0.20,
    shadow_chroma_tol: float = 14.0,
    bg_threshold: int = 30,
) -> Optional[np.ndarray]:
    """
    Unified camera-capture widget.

    • On the Pi (picamera2 available): shows "Capture preview" + "Use this frame"
      buttons that read directly from PiCameraStream.  No browser permissions needed.
    • Elsewhere: falls back to st.camera_input() with an HTTPS/localhost note.

    Returns a confirmed (H,W,3) uint8 RGB array, or None until the user confirms.
    """
    preview_key = f"_prev_{key_prefix}"
    cam = _open_pi_camera()

    def _processed_diff(frm: np.ndarray, bg: np.ndarray) -> np.ndarray:
        """Return a uint8 grayscale diff after contrast boost + shadow removal."""
        f = _apply_clahe(frm, clahe_clip) if contrast_enhance else frm
        b = _apply_clahe(bg,  clahe_clip) if contrast_enhance else bg
        diff = np.abs(f.astype(np.int16) - b.astype(np.int16)).mean(2).astype(np.uint8)
        if shadow_removal:
            # Zero out pixels that are classified as shadows so they appear dark.
            shad = remove_shadows(
                diff > bg_threshold, frm, bg,
                darkness=shadow_darkness, chroma_tol=shadow_chroma_tol,
            )
            diff = (diff * shad.astype(np.uint8))
        return diff

    if cam is not None:
        # ── Pi camera path ────────────────────────────────────────────
        cap_col, conf_col = st.columns(2)

        with cap_col:
            if st.button("📷 Capture preview", key=f"btn_prev_{key_prefix}",
                         use_container_width=True):
                try:
                    st.session_state[preview_key] = cam.read_rgb()
                except Exception as e:
                    st.error(f"Camera read failed: {e}")

        has_preview = preview_key in st.session_state
        with conf_col:
            if st.button("✅ Use this frame", key=f"btn_use_{key_prefix}",
                         disabled=not has_preview, type="primary",
                         use_container_width=True):
                frame = st.session_state.pop(preview_key)
                return frame

        if has_preview:
            frame = st.session_state[preview_key]
            if background is not None and frame.shape == background.shape:
                prev_c, diff_c = st.columns(2)
                prev_c.image(frame,
                             caption="Preview — click 'Use this frame' to confirm",
                             use_container_width=True)
                diff_c.image(_processed_diff(frame, background),
                             caption="Difference from background (processed)",
                             use_container_width=True, clamp=True)
            else:
                st.image(frame, caption="Preview — click 'Use this frame' to confirm",
                         use_container_width=True)
        else:
            st.info(f"Click **Capture preview** to photograph: *{label}*")

    else:
        # ── Browser-camera fallback ───────────────────────────────────
        st.caption(
            "ℹ️ Pi camera not detected.  "
            "Browser camera requires **HTTPS** or **localhost** access.  "
            "If running over the local network, open `http://localhost:8501` "
            "on the Pi itself, or configure Streamlit with an SSL certificate."
        )
        photo = st.camera_input(label, key=f"webcam_{key_prefix}")
        if photo is not None:
            st.session_state[preview_key] = np.array(Image.open(photo).convert("RGB"))

        has_preview = preview_key in st.session_state
        if has_preview:
            if background is not None:
                frame = st.session_state[preview_key]
                if frame.shape == background.shape:
                    prev_c, diff_c = st.columns(2)
                    prev_c.image(frame, caption="Captured", use_container_width=True)
                    diff_c.image(_processed_diff(frame, background),
                                 caption="Difference from background (processed)",
                                 use_container_width=True, clamp=True)
            if st.button("✅ Use this frame", key=f"btn_use_{key_prefix}",
                         type="primary", use_container_width=True):
                return st.session_state.pop(preview_key)

    return None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_shape_masks(
    shape_type: str,
    angles: list,
    H: int, W: int,
    cx: int, cy: int,
    fx_syn: float,
    cam_dist: float,
    r_m: float,
    theta_rad: float = np.pi / 2,
) -> list:
    """
    Generate synthetic 2D silhouette masks by projecting a 3D voxel shape at
    each viewpoint using the full hemispherical camera model from reconstruct.py.

    theta_rad: polar angle from zenith (pi/2 = horizontal side view).
    shape_type: "brick" | "L" | "torus"
    """
    from gripper_cv.heapgrasp.reconstruct import _view_transform

    V = 40
    lin = np.linspace(-1.0, 1.0, V)
    xi, yi, zi = np.meshgrid(lin, lin, lin, indexing='ij')

    if shape_type == "brick":
        occ = (np.abs(xi) <= 0.90) & (np.abs(yi) <= 0.65) & (np.abs(zi) <= 0.45)
    elif shape_type == "L":
        arm_h = (np.abs(xi) <= 0.90) & (np.abs(zi) <= 0.28) & (np.abs(yi) <= 0.65)
        arm_v = (xi >= 0.45) & (np.abs(zi) <= 0.90) & (np.abs(yi) <= 0.65)
        occ = arm_h | arm_v
    elif shape_type == "torus":
        # Ring standing upright in XY plane (hole along Z).
        # At theta=pi/2, 0° looks like a ring; at 90° collapses to a rectangle.
        r_maj, r_min = 0.62, 0.33
        dist = np.sqrt((np.sqrt(xi**2 + yi**2) - r_maj)**2 + zi**2)
        occ = dist <= r_min
    else:
        raise ValueError(f"Unknown shape_type: {shape_type!r}")

    ix, iy, iz = np.where(occ)
    X = lin[ix] * r_m
    Y = lin[iy] * r_m
    Z = lin[iz] * r_m
    pts = np.stack([X, Y, Z], axis=0)  # (3, N)

    K_syn = np.array([[fx_syn, 0, cx], [0, fx_syn, cy], [0, 0, 1]], dtype=np.float64)

    masks_out = []
    for a in angles:
        R, C = _view_transform(a, theta_rad, cam_dist)
        pts_cam = R @ (pts - C[:, None])  # (3, N)
        depth = pts_cam[2]

        proj = K_syn @ pts_cam
        u = np.round(proj[0] / (depth + 1e-9)).astype(int)
        v = np.round(proj[1] / (depth + 1e-9)).astype(int)

        mask = np.zeros((H, W), dtype=bool)
        valid = (depth > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        uv, vv = u[valid], v[valid]

        for uc in np.unique(uv):
            rows = vv[uv == uc]
            mask[rows.min():rows.max() + 1, uc] = True

        masks_out.append(mask)

    return masks_out


@st.cache_data(show_spinner=False)
def _run_sfs(
    masks_bytes: Tuple[bytes, ...],
    shapes: Tuple[Tuple[int, int], ...],
    angles: Tuple[float, ...],
    V: int, diameter: float, cam_dist: float, fov: float,
    theta_rad: float = np.pi / 2,
) -> Tuple[np.ndarray, np.ndarray]:
    masks = [np.frombuffer(b, np.uint8).reshape(s).astype(bool)
             for b, s in zip(masks_bytes, shapes)]
    H, W = shapes[0] if shapes else (480, 640)
    K = default_camera_matrix(W, H, fov_deg=fov)
    thetas = [theta_rad] * len(masks)
    center = estimate_grid_center(masks, K, cam_dist,
                                  thetas_rad=thetas, angles_deg=list(angles))
    voxels = shape_from_silhouette(masks, list(angles), K,
                                   volume_size=V, object_diameter=diameter,
                                   camera_distance=cam_dist, grid_center=center,
                                   thetas_rad=thetas)
    return voxels, center


def generate_grasp_plan(
    voxels: np.ndarray,
    diameter: float,
    cam_dist: float,
    n_views: int,
) -> str:
    return format_grasp_plan(voxels, diameter, cam_dist, n_views)


def _overlay_mask(img: np.ndarray, mask: np.ndarray,
                  color=(50, 220, 100), alpha: float = 0.45) -> np.ndarray:
    out = img.astype(np.float32)
    c   = np.array(color, dtype=np.float32)
    out[mask] = out[mask] * (1 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _pca(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = pts.mean(0)
    cov = np.cov((pts - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return centroid, eigvals[order], eigvecs[:, order]


def _ply_bytes(voxels: np.ndarray, diameter: float, grid_center: np.ndarray | None = None) -> bytes:
    pts = voxels_to_pointcloud(voxels, diameter, grid_center)
    buf = io.BytesIO()
    buf.write(b"ply\nformat ascii 1.0\n")
    buf.write(f"element vertex {len(pts)}\n".encode())
    buf.write(b"property float x\nproperty float y\nproperty float z\nend_header\n")
    for p in pts:
        buf.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n".encode())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Grasp plan generator
# ---------------------------------------------------------------------------

def generate_grasp_plan(
    voxels: np.ndarray,
    diameter: float,
    cam_dist: float,
    n_views: int,
    grid_center: np.ndarray | None = None,
) -> str:
    """
    Generate a natural-language arm movement plan from a reconstructed voxel grid.

    Coordinate convention (camera = gripper tip):
      X  →  right in camera frame  (positive = translate gripper right)
      Y  →  up in camera frame     (positive = translate gripper up)
      Z  →  forward (optical axis) (approach direction)
    """
    pts = voxels_to_pointcloud(voxels, diameter, grid_center)
    if len(pts) < 8:
        return ("⚠  Insufficient point cloud data.\n"
                "Capture more views for a reliable grasp plan.")

    centroid, eigvals, eigvecs = _pca(pts)

    # Physical extents (2 × 2.5σ per principal axis), sorted largest first
    spans_m = sorted(2 * np.sqrt(np.maximum(eigvals, 0)) * 2.5, reverse=True)
    dim_str = "  ×  ".join(f"{d * 100:.1f} cm" for d in spans_m)

    # Lateral offsets: how far is the object centroid from the camera optical axis?
    dx_cm = float(centroid[0]) * 100   # +right, −left
    dy_cm = float(centroid[1]) * 100   # +up,   −down

    # Jaw axis = longest PCA axis (jaws open/close along this)
    jaw_ax = eigvecs[:, 0]

    # Grasp approach axis = shortest PCA axis (gripper approaches along this)
    grasp_ax = eigvecs[:, 2]

    # Jaw opening = span along jaw axis + safety clearance
    jaw_span_cm = float(np.sqrt(max(eigvals[0], 0))) * 5 * 100   # ±2.5σ
    jaw_open_cm = jaw_span_cm + 1.5

    # Wrist rotation: angle between jaw_ax projected onto image plane and camera X
    jaw_xy  = np.array([float(jaw_ax[0]), float(jaw_ax[1])])
    wrist_deg = float(np.degrees(np.arctan2(jaw_xy[1], jaw_xy[0]))) if np.linalg.norm(jaw_xy) > 0.1 else 0.0

    # Descriptive strings
    _axis_desc  = {0: "horizontal (left-right)", 1: "vertical (up-down)", 2: "axial (depth)"}
    dominant_jaw   = int(np.argmax(np.abs(jaw_ax)))
    dominant_grasp = int(np.argmax(np.abs(grasp_ax)))
    jaw_desc    = _axis_desc[dominant_jaw]
    grasp_desc  = _axis_desc[dominant_grasp]

    if spans_m[0] > 0.10:
        grip_type = "two-finger power grip (large object)"
    elif spans_m[0] < 0.04:
        grip_type = "precision pinch (small object)"
    else:
        grip_type = "standard parallel-jaw pinch"

    fill_pct = 100.0 * int(voxels.sum()) / voxels.size
    W = 65
    SEP = "═" * W
    sec = "─" * (W - 4)

    def _row(label, value, indent=2):
        return f"{'':>{indent}}{label:<30}{value}"

    lines = [
        SEP,
        "  THEORETICAL GRASP PLAN  —  Ford Industrial Arm",
        f"  {n_views}-view HEAPGrasp reconstruction  |  camera distance {cam_dist*100:.0f} cm",
        SEP,
        "",
        "  OBJECT SUMMARY",
        f"  {'Estimated dimensions':<28}{dim_str}",
        f"  {'Jaw span axis':<28}{jaw_desc}",
        f"  {'Approach axis':<28}{grasp_desc}",
        f"  {'Recommended grip':<28}{grip_type}",
        f"  {'Reconstruction fill':<28}{fill_pct:.0f} %  ({int(voxels.sum()):,} occupied voxels)",
        "",
        f"  {sec}",
        "  ①  LATERAL ALIGNMENT",
        f"  {sec}",
        "  Camera (mounted at gripper tip) sees the object at:",
    ]

    if abs(dx_cm) > 0.3:
        dirn = "right" if dx_cm > 0 else "left"
        lines.append(f"    • {abs(dx_cm):.1f} cm to the {dirn} of optical centre")
        lines.append(f"      → Translate TCP {abs(dx_cm):.1f} cm {'right (+X)' if dx_cm > 0 else 'left (−X)'}")
    else:
        lines.append("    • Centred horizontally  (no X correction needed)")

    if abs(dy_cm) > 0.3:
        dirn = "above" if dy_cm > 0 else "below"
        lines.append(f"    • {abs(dy_cm):.1f} cm {dirn} optical centre")
        lines.append(f"      → Translate TCP {abs(dy_cm):.1f} cm {'up (+Y)' if dy_cm > 0 else 'down (−Y)'}")
    else:
        lines.append("    • Centred vertically    (no Y correction needed)")

    lines += [
        "",
        f"  {sec}",
        "  ②  WRIST ROTATION",
        f"  {sec}",
    ]
    if abs(wrist_deg) > 5.0:
        dirn = "counter-clockwise" if wrist_deg > 0 else "clockwise"
        lines += [
            f"  Jaw axis is {abs(wrist_deg):.0f}° from horizontal in the image plane.",
            f"    → Rotate wrist {abs(wrist_deg):.0f}°  {dirn}  (align jaws with object long axis)",
        ]
    else:
        lines.append("  Jaw axis is approximately horizontal — no wrist rotation needed.")

    lines += [
        "",
        f"  {sec}",
        "  ③  OPEN JAWS",
        f"  {sec}",
        f"  Object span along jaw axis:  {jaw_span_cm:.1f} cm",
        f"    → Set jaw opening to  {jaw_open_cm:.1f} cm",
        f"       ({jaw_span_cm:.1f} cm object  +  1.5 cm clearance)",
        "",
        f"  {sec}",
        "  ④  APPROACH",
        f"  {sec}",
        f"  Object is {cam_dist * 100:.0f} cm from the gripper tip (along optical axis).",
        f"    → Advance {cam_dist * 100:.0f} cm along gripper Z-axis.",
        f"    → Speed: SLOW  —  transparent/specular surface, uncertain contact geometry.",
        "",
        f"  {sec}",
        "  ⑤  GRASP",
        f"  {sec}",
        "    → Close jaws to contact resistance.",
        "    → Target force: LIGHT.  Do not over-squeeze transparent objects.",
        "    → Monitor SlipDetectorCNN output:",
        "         stable  →  continue",
        "         slipping  →  pause, re-seat object, or abort mission.",
        "",
        f"  {sec}",
        "  ⑥  RETRIEVE",
        f"  {sec}",
        "    → Raise TCP 5 cm vertically to clear the turntable surface.",
        "    → Translate to deposit location at REDUCED speed.",
        "    → Confirm stable hold before increasing speed.",
        "",
        SEP,
        "  ⚠  SAFETY NOTES",
        "",
        "  1. This plan is THEORETICAL — generated from camera data only.",
        "  2. No real arm commands have been issued.",
        "  3. A human operator must verify all clearances before executing.",
        "  4. Reconstruction accuracy improves with more views.",
        f"     Current quality: {fill_pct:.0f}% voxel fill from {n_views} views.",
        SEP,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3D figure builder
# ---------------------------------------------------------------------------

def _make_scan_figure(
    voxels: np.ndarray,
    diameter: float,
    cam_dist: float,
    visited_angles: Optional[List[float]] = None,
    nbv_angles: Optional[List[float]] = None,
    show_grasp: bool = True,
    max_pts: int = 8000,
    grid_center: np.ndarray | None = None,
    grasps: Optional[List[GraspCandidate]] = None,
):
    import plotly.graph_objects as go

    pts = voxels_to_pointcloud(voxels, diameter, grid_center)
    if len(pts) == 0:
        return go.Figure()

    rng = np.random.default_rng(0)
    if len(pts) > max_pts:
        pts = pts[rng.choice(len(pts), max_pts, replace=False)]

    traces: list = []

    z_range = float(pts[:, 1].max() - pts[:, 1].min()) + 1e-9
    z_norm  = (pts[:, 1] - pts[:, 1].min()) / z_range
    traces.append(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 2], z=pts[:, 1],
        mode="markers",
        marker=dict(size=2.5, color=z_norm, colorscale="Plasma", opacity=0.75,
                    colorbar=dict(title="Height", thickness=12, len=0.45, x=1.02)),
        name="Object surface",
    ))

    if len(pts) >= 4:
        centroid, eigvals, eigvecs = _pca(pts)
        ax_colors = ["#ef4444", "#22c55e", "#3b82f6"]
        ax_labels = ["Long axis", "Mid axis", "Grasp axis"]
        spans = np.sqrt(np.maximum(eigvals, 0)) * 2.5

        for i in range(3):
            v  = eigvecs[:, i]
            p1 = centroid + v * spans[i]
            p2 = centroid - v * spans[i]
            traces.append(go.Scatter3d(
                x=[p1[0], centroid[0], p2[0]],
                y=[p1[2], centroid[2], p2[2]],
                z=[p1[1], centroid[1], p2[1]],
                mode="lines",
                line=dict(color=ax_colors[i], width=4),
                name=ax_labels[i],
            ))

        if show_grasp and grasps:
            grasp_colors = ["#22c55e", "#fbbf24", "#f97316", "#a78bfa", "#60a5fa"]
            for rank, g in enumerate(grasps):
                col = grasp_colors[rank % len(grasp_colors)]
                label_prefix = f"#{rank+1} (score {g.score:.2f})"

                # Line between contacts = jaw opening
                traces.append(go.Scatter3d(
                    x=[g.contact_1[0], g.contact_2[0]],
                    y=[g.contact_1[2], g.contact_2[2]],
                    z=[g.contact_1[1], g.contact_2[1]],
                    mode="lines+markers",
                    line=dict(color=col, width=6),
                    marker=dict(size=6, color=col, symbol="circle"),
                    name=f"Grasp {label_prefix}",
                ))

                # Approach arrow: centre → centre + approach_dir * 0.04
                arr_end = g.center + g.approach_dir * 0.04
                traces.append(go.Scatter3d(
                    x=[g.center[0], arr_end[0]],
                    y=[g.center[2], arr_end[2]],
                    z=[g.center[1], arr_end[1]],
                    mode="lines",
                    line=dict(color=col, width=3, dash="dot"),
                    showlegend=False,
                ))

                # TCP centre marker
                traces.append(go.Scatter3d(
                    x=[g.center[0]], y=[g.center[2]], z=[g.center[1]],
                    mode="markers",
                    marker=dict(size=7, symbol="cross", color=col),
                    showlegend=False,
                ))

    ring_t = np.linspace(0, 2 * np.pi, 128)
    traces.append(go.Scatter3d(
        x=cam_dist * np.sin(ring_t), y=cam_dist * np.cos(ring_t), z=np.zeros(128),
        mode="lines",
        line=dict(color="rgba(148,163,184,0.25)", width=1, dash="dot"),
        showlegend=False,
    ))

    if visited_angles:
        vx = [cam_dist * np.sin(np.radians(a)) for a in visited_angles]
        vy = [cam_dist * np.cos(np.radians(a)) for a in visited_angles]
        traces.append(go.Scatter3d(
            x=vx, y=vy, z=[0.0] * len(visited_angles),
            mode="markers+text",
            marker=dict(size=8, color="#60a5fa", symbol="square"),
            text=[f"{a:.0f}°" for a in visited_angles],
            textposition="top center",
            name="📷 Captured views",
        ))

    if nbv_angles:
        nx = [cam_dist * np.sin(np.radians(a)) for a in nbv_angles]
        ny = [cam_dist * np.cos(np.radians(a)) for a in nbv_angles]
        traces.append(go.Scatter3d(
            x=nx, y=ny, z=[0.0] * len(nbv_angles),
            mode="markers+text",
            marker=dict(size=8, color="#fb923c", symbol="diamond"),
            text=[f"▶{a:.0f}°" for a in nbv_angles],
            textposition="top center",
            name="🎯 NBV suggestions",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Z (m)", zaxis_title="Y (m)",
            aspectmode="data",
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#313244"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#313244"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#313244"),
        ),
        legend=dict(font=dict(size=11), itemsizing="constant",
                    bgcolor="rgba(30,30,46,0.85)", bordercolor="#45475a"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=640,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ===========================================================================
# Page 1 — Overview
# ===========================================================================

if page == "🏠 Overview":
    st.title("🦾 HEAPGrasp — AI Gripper Vision System")
    st.markdown(
        "End-to-end transparent-object grasping pipeline on "
        "**Raspberry Pi 5 + Hailo-8L AI Hat+**. "
        "Designed for mounting at the end of the Ford industrial arm — "
        "no depth camera required."
    )
    st.info(
        "**How the demo works:** Mount the Pi + camera at the gripper tip. "
        "Place the target object on a turntable in front of the gripper. "
        "Use **🎬 Live Capture** to photograph it from N angles, "
        "then **📦 3D Object Scan** reconstructs the shape and outputs "
        "a step-by-step plan telling the arm how to grasp it."
    )

    st.markdown("---")
    st.subheader("Pipeline")
    cols = st.columns([1, .1, 1, .1, 1, .1, 1, .1, 1])
    steps = [
        ("📷", "Capture", "Turntable: N views from the mounted camera"),
        ("✂️", "Segment", "BG subtraction or DeepLabv3+ silhouettes"),
        ("📐", "Calibrate", "ArUco → camera distance; silhouette → object size"),
        ("🧊", "Reconstruct", "Shape from Silhouette — voxel carving"),
        ("🎯", "Plan", "NBV + grasp plan → arm movement instructions"),
    ]
    for ci, (icon, title, desc) in enumerate(steps):
        with cols[ci * 2]:
            st.markdown(f"""
<div class='pipeline-step'>
  <div style='font-size:2em'>{icon}</div>
  <strong>{title}</strong><br>
  <small style='opacity:.7'>{desc}</small>
</div>""", unsafe_allow_html=True)
        if ci < len(steps) - 1:
            with cols[ci * 2 + 1]:
                st.markdown(
                    "<div style='text-align:center;font-size:1.4em;"
                    "padding-top:24px'>→</div>",
                    unsafe_allow_html=True)

    st.markdown("---")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Platform", "Raspberry Pi 5")
    sc2.metric("NPU", "Hailo-8L · 13 TOPS" if _HAILO_OK else "Hailo-8L · offline")
    sc3.metric("Camera", "Pi Cam Module 3")
    sc4.metric("HailoRT", _HAILO_VERSION if _HAILO_VERSION else "—")

    st.markdown("---")
    st.subheader("Models")
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, (name, task, inp, note, icon) in zip([m1, m2, m3, m4, m5], [
        ("MobileNetV3-S",    "Classification",    "224×224",     "~2 ms NPU",         "🔍"),
        ("DeepLabv3+",       "Segmentation",      "512×512",     "~60 ms NPU",        "✂️"),
        ("DeepLabv3+ FT",    "Transparent seg.",  "512×512",     "4-class fine-tune", "🪟"),
        ("SlipDetectorCNN",  "Tactile slip",      "128×128",     "3 conv + 2 FC",     "🤏"),
        ("NBV Planner",      "Active perception", "Voxel grid",  "Pure numpy",        "🎯"),
    ]):
        with col:
            st.markdown(f"**{icon} {name}**")
            st.caption(f"Task: {task}")
            st.caption(f"Input: {inp}")
            st.caption(note)

# ===========================================================================
# Page 2 — Live Capture
# ===========================================================================

elif page == "🎬 Live Capture":
    st.title("🎬 Live Capture — Guided Session")
    st.markdown(
        "Mount the camera at the gripper tip and point it at the object. "
        "The arm sweeps around the object on a hemisphere — set the **polar angle θ** "
        "below to match how your rig is positioned, then capture background + N views."
    )

    # Camera status banner
    if _PI_CAMERA_AVAILABLE and _open_pi_camera() is not None:
        st.success("✅ Pi camera connected — using direct PiCameraStream capture.")
    elif _PI_CAMERA_AVAILABLE:
        st.error(
            "Pi camera module detected but failed to open. "
            "Check the ribbon cable and that `libcamera` can see the camera "
            "(`libcamera-hello --list-cameras`)."
        )
    else:
        st.warning(
            "Pi camera not detected.  "
            "Falling back to **browser camera** — this requires the page to be served "
            "over **HTTPS** or opened as `http://localhost:8501` directly on the Pi.  \n"
            "To access via the local network with camera support, either:  \n"
            "- SSH tunnel: `ssh -L 8501:localhost:8501 pi@<ip>` then open `http://localhost:8501`  \n"
            "- Or configure Streamlit with an SSL cert in `~/.streamlit/config.toml`."
        )

    st.markdown("---")

    # Per-session settings
    cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns([1, 1, 2, 2])
    with cfg_col1:
        n_views = st.number_input("Views", min_value=4, max_value=24,
                                  value=st.session_state.cap_n_views, step=1)
        st.session_state.cap_n_views = int(n_views)
    with cfg_col2:
        step_deg = 360.0 / n_views
        st.metric("Step per rotation", f"{step_deg:.1f}°")
    with cfg_col3:
        theta_input = st.slider(
            "Camera polar angle θ from zenith",
            min_value=0, max_value=90,
            value=int(round(st.session_state.cap_theta_deg)),
            help=(
                "**90°** = camera at the same height as the object, looking horizontally "
                "(classic turntable setup).\n\n"
                "**36° (π/5)** = HEAPGrasp hand-eye default — arm holds camera above "
                "and slightly in front, looking down.\n\n"
                "**0°** = camera directly above, looking straight down."
            ),
        )
        st.session_state.cap_theta_deg = float(theta_input)
        st.caption(f"θ = {theta_input}° = {np.radians(theta_input):.3f} rad — "
                   f"camera is {90 - theta_input}° above horizontal")
    with cfg_col4:
        if st.button("🗑️ Reset session", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k.startswith("_prev_"):
                    del st.session_state[k]
            st.session_state.cap_bg       = None
            st.session_state.cap_frames   = []
            st.session_state.cap_angles   = []
            st.session_state.cap_theta_deg = 90.0
            st.rerun()

    n_done = len(st.session_state.cap_frames)
    has_bg = st.session_state.cap_bg is not None
    total_steps = n_views + 1
    steps_done  = (1 if has_bg else 0) + n_done
    st.progress(steps_done / total_steps,
                text=f"Step {steps_done} / {total_steps} complete")

    with st.expander("✨ Preview contrast & shadow settings", expanded=False):
        cap_con, cap_shad = st.columns(2)
        with cap_con:
            st.markdown("**Contrast (CLAHE)**")
            cap_contrast_on  = st.toggle("Enable contrast boost", value=False, key="cap_contrast_on")
            cap_clahe_clip   = st.slider("CLAHE clip limit", 1.0, 8.0, 2.0, 0.5,
                                         key="cap_clahe_clip",
                                         help="Higher = stronger local contrast boost.")
        with cap_shad:
            st.markdown("**Shadow removal**")
            cap_shadow_on    = st.toggle("Enable shadow removal", value=True, key="cap_shadow_on")
            cs1, cs2, cs3   = st.columns(3)
            cap_bg_thresh    = cs1.slider("BG threshold", 5, 80, 30, 5, key="cap_bg_thresh")
            cap_sh_dark      = cs2.slider("Darkness", 0.05, 0.50, 0.20, 0.05,
                                          key="cap_sh_dark",
                                          help="Min L* drop to classify as shadow.")
            cap_sh_chroma    = cs3.slider("Chroma tol.", 2.0, 30.0, 14.0, 1.0,
                                          key="cap_sh_chroma",
                                          help="Max |Δa*|/|Δb*| for a shadow pixel.")

    _cap_kwargs = dict(
        contrast_enhance=cap_contrast_on,
        clahe_clip=cap_clahe_clip,
        shadow_removal=cap_shadow_on,
        shadow_darkness=cap_sh_dark,
        shadow_chroma_tol=cap_sh_chroma,
        bg_threshold=cap_bg_thresh,
    )

    st.markdown("---")

    # ── Step 1: background ────────────────────────────────────────────
    with st.expander(
        f"{'✅' if has_bg else '👉'} Step 1 — Background (no object)",
        expanded=not has_bg,
    ):
        if has_bg:
            st.image(st.session_state.cap_bg, caption="Background (saved)",
                     use_container_width=True)
        else:
            st.markdown(
                "Remove the object from the turntable. "
                "Keep lighting and camera position exactly as during the scan."
            )
            bg_frame = _pi_capture_ui("Background (no object)", "bg", **_cap_kwargs)
            if bg_frame is not None:
                st.session_state.cap_bg = bg_frame
                st.rerun()

    # ── Steps 2..N+1: object views ────────────────────────────────────
    if has_bg:
        for view_i in range(n_views):
            angle = view_i * step_deg
            done  = view_i < n_done

            with st.expander(
                f"{'✅' if done else ('👉' if view_i == n_done else '  ')}"
                f" View {view_i + 1}/{n_views}  —  {angle:.0f}°",
                expanded=(view_i == n_done),
            ):
                if done:
                    st.image(st.session_state.cap_frames[view_i],
                             caption=f"Captured at {angle:.0f}°",
                             use_container_width=True)
                elif view_i == n_done:
                    st.markdown(
                        f"Rotate the object to **{angle:.0f}°** "
                        f"({step_deg:.0f}° from the previous position)."
                    )
                    frame = _pi_capture_ui(
                        f"View {view_i + 1} at {angle:.0f}°",
                        f"view_{view_i}",
                        background=st.session_state.cap_bg,
                        **_cap_kwargs,
                    )
                    if frame is not None:
                        bg = st.session_state.cap_bg
                        if frame.shape != bg.shape:
                            frame = np.array(
                                Image.fromarray(frame).resize(
                                    (bg.shape[1], bg.shape[0]), Image.LANCZOS
                                )
                            )
                        st.session_state.cap_frames.append(frame)
                        st.session_state.cap_angles.append(angle)
                        st.rerun()
                else:
                    st.caption("Waiting for previous views.")

    # ── Completion banner ─────────────────────────────────────────────
    if n_done == n_views and has_bg:
        st.markdown("---")
        st.success(
            f"✅ **Session complete!**  {n_views} views captured.  "
            "Go to **📦 3D Object Scan** to reconstruct and get the grasp plan."
        )
        st.markdown("**Captured frames:**")
        thumb_cols = st.columns(min(n_views, 8))
        for i, (col, frame) in enumerate(
            zip(thumb_cols, st.session_state.cap_frames[:8])
        ):
            with col:
                st.image(frame, caption=f"{st.session_state.cap_angles[i]:.0f}°",
                         use_container_width=True)

# ===========================================================================
# Page 3 — Image Classifier
# ===========================================================================

elif page == "🔍 Image Classifier":
    st.title("Image Classifier — MobileNetV3-Small")
    st.markdown(
        "Upload any image or use the camera. "
        "MobileNetV3-Small (ImageNet pretrained) returns the top-5 classes. "
        "On the Hailo-8L this runs at **~2 ms**."
    )

    if _HAILO_OK:
        st.success(
            f"✅ Hailo-8L detected (HailoRT {_HAILO_VERSION}). "
            "Compile MobileNetV3 ONNX to HEF with `gripper-cv-export-onnx` to use the NPU."
        )
    else:
        st.info("Hailo-8L not detected — inference runs on CPU.")

    input_src = st.radio("Image source", ["Upload file", "Camera"], horizontal=True)
    image_pil = None

    if input_src == "Upload file":
        uploaded = st.file_uploader("Upload an image",
                                    type=["jpg", "jpeg", "png", "bmp"])
        if uploaded:
            image_pil = Image.open(uploaded).convert("RGB")
    else:
        cam_snap = st.camera_input("📷 Take a photo to classify")
        if cam_snap:
            image_pil = Image.open(cam_snap).convert("RGB")

    if image_pil is not None:
        col_img, col_res = st.columns([1, 1])
        with col_img:
            st.image(image_pil, caption="Input", use_container_width=True)
        with col_res:
            with st.spinner("Running MobileNetV3-Small…"):
                try:
                    import torch
                    from torchvision import models

                    weights = models.MobileNet_V3_Small_Weights.DEFAULT
                    model   = models.mobilenet_v3_small(weights=weights).eval()
                    tensor  = weights.transforms()(image_pil).unsqueeze(0)
                    with torch.inference_mode():
                        probs = torch.softmax(model(tensor), dim=1)[0]

                    top5_probs, top5_idx = torch.topk(probs, 5)
                    meta = weights.meta["categories"]

                    st.markdown("#### Top-5 predictions")
                    for rank, (prob, idx) in enumerate(
                        zip(top5_probs.tolist(), top5_idx.tolist()), 1
                    ):
                        label = meta[idx] if idx < len(meta) else f"cls_{idx}"
                        st.progress(float(prob), text=f"**#{rank} {label}** — {prob:.1%}")

                    import plotly.graph_objects as go
                    labels = [(meta[i] if i < len(meta) else f"cls{i}")[:28]
                              for i in top5_idx.tolist()]
                    fig = go.Figure(go.Bar(
                        x=top5_probs.tolist()[::-1],
                        y=labels[::-1],
                        orientation="h",
                        marker_color=["#a6e3a1","#94e2d5","#89dceb","#74c7ec","#89b4fa"][::-1],
                        text=[f"{p:.1%}" for p in top5_probs.tolist()[::-1]],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        xaxis=dict(range=[0, 1], tickformat=".0%",
                                   gridcolor="#313244"),
                        margin=dict(l=0, r=60, t=10, b=0),
                        height=220,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                except ImportError:
                    st.warning(
                        "PyTorch / torchvision not installed.\n\n"
                        "```\npip install torch torchvision "
                        "--index-url https://download.pytorch.org/whl/cpu\n```"
                    )
    else:
        st.info("Take a photo or upload an image above.")

# ===========================================================================
# Page 4 — Segmentation
# ===========================================================================

elif page == "✂️ Segmentation":
    st.title("Silhouette Extraction")

    tab_bg, tab_dr = st.tabs(["Background Subtraction", "Domain Randomization Preview"])

    with tab_bg:
        st.markdown(
            "Provide a **background** frame and an **object** frame — "
            "or use the cameras below. Adjust the threshold to see the mask and overlay."
        )
        col_bg, col_obj = st.columns(2)
        with col_bg:
            src = st.radio("Background source", ["Upload", "Camera"], horizontal=True,
                           key="bg_src")
            if src == "Upload":
                bg_file = st.file_uploader("Background", type=["jpg","jpeg","png"], key="bg_up")
                bg_rgb  = np.array(Image.open(bg_file).convert("RGB")) if bg_file else None
            else:
                bg_cam = st.camera_input("📷 Background (no object)", key="bg_cam")
                bg_rgb = np.array(Image.open(bg_cam).convert("RGB")) if bg_cam else None

        with col_obj:
            src2 = st.radio("Object source", ["Upload", "Camera"], horizontal=True,
                            key="obj_src")
            if src2 == "Upload":
                obj_file = st.file_uploader("Object frame", type=["jpg","jpeg","png"], key="obj_up")
                obj_rgb  = np.array(Image.open(obj_file).convert("RGB")) if obj_file else None
            else:
                obj_cam = st.camera_input("📷 Object frame", key="obj_cam")
                obj_rgb = np.array(Image.open(obj_cam).convert("RGB")) if obj_cam else None

        threshold = st.slider("Pixel-difference threshold", 5, 80, 30, 5)

        with st.expander("✨ Contrast & shadow removal", expanded=False):
            con1, con2 = st.columns(2)
            with con1:
                st.markdown("**Contrast (CLAHE)**")
                contrast_on  = st.toggle("Enable contrast boost", value=False, key="seg_contrast_on")
                clahe_clip_v = st.slider("CLAHE clip limit", 1.0, 8.0, 2.0, 0.5,
                                         key="seg_clahe_clip",
                                         help="Higher = stronger local contrast boost. "
                                              "2–4 is typical; above 6 may introduce artefacts.")
            with con2:
                st.markdown("**Shadow removal**")
                shadow_on  = st.toggle("Enable shadow removal", value=True, key="seg_shadow_on")
                sh1, sh2   = st.columns(2)
                sh_dark    = sh1.slider("Darkness", 0.05, 0.50, 0.20, 0.05,
                                        key="seg_sh_dark",
                                        help="Min L* drop to flag as shadow.")
                sh_chroma  = sh2.slider("Chroma tol.", 2.0, 30.0, 14.0, 1.0,
                                        key="seg_sh_chroma",
                                        help="Max |Δa*|/|Δb*| for a shadow pixel.")

        def _show_seg(bg, obj):
            if bg.shape != obj.shape:
                obj = np.array(
                    Image.fromarray(obj).resize((bg.shape[1], bg.shape[0]), Image.LANCZOS)
                )
            sess = CaptureSession(frames=[obj], angles_deg=[0.0], background=bg)
            mask = extract_silhouettes(
                sess, method="background", bg_threshold=threshold,
                shadow_removal=shadow_on, shadow_darkness=sh_dark,
                shadow_chroma_tol=sh_chroma,
                contrast_enhance=contrast_on, clahe_clip=clahe_clip_v,
            )[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.image(bg,  caption="Background",   use_container_width=True)
            c2.image(obj, caption="Object frame",  use_container_width=True)
            c3.image(mask.astype(np.uint8)*255, caption=f"Mask ({mask.sum():,} px)",
                     use_container_width=True, clamp=True)
            c4.image(_overlay_mask(obj, mask), caption="Overlay", use_container_width=True)
            a, b = st.columns(2)
            a.metric("Foreground coverage", f"{mask.mean()*100:.1f}%")
            b.metric("Foreground pixels",    f"{int(mask.sum()):,}")

        if bg_rgb is not None and obj_rgb is not None:
            _show_seg(bg_rgb, obj_rgb)
        else:
            st.markdown("**Synthetic demo** (no images provided)")
            H, W = 240, 320
            bg  = np.full((H, W, 3), 35, dtype=np.uint8)
            obj = bg.copy()
            cv2.ellipse(obj, (W//2, H//2), (90, 65), 20, 0, 360, (160, 160, 165), -1)
            cv2.ellipse(obj, (W//2-20, H//2-20), (22, 15), 20, 0, 360, (230, 230, 235), -1)
            _show_seg(bg, obj)

    with tab_dr:
        st.markdown(
            "**Domain randomization** augments training images to close the sim-to-real gap. "
            "Adjust probabilities and see the effect on N samples."
        )
        dr_src = st.radio("Source image", ["Synthetic", "Upload", "Camera"], horizontal=True)
        if dr_src == "Upload":
            dr_file = st.file_uploader("Source image", type=["jpg","jpeg","png"], key="dr_up")
            src_rgb = (np.array(Image.open(dr_file).convert("RGB").resize((320, 240)))
                       if dr_file else None)
        elif dr_src == "Camera":
            dr_cam  = st.camera_input("📷 Source image", key="dr_cam")
            src_rgb = (np.array(Image.open(dr_cam).convert("RGB").resize((320, 240)))
                       if dr_cam else None)
        else:
            src_rgb = np.full((240, 320, 3), 40, dtype=np.uint8)
            cv2.ellipse(src_rgb, (160, 120), (85, 60), 0, 0, 360, (140, 180, 200), -1)

        if src_rgb is not None:
            p1c, p2c, p3c, p4c = st.columns(4)
            p_bg     = p1c.slider("P(bg replace)",    0.0, 1.0, 0.8, 0.1)
            p_jitter = p2c.slider("P(colour jitter)", 0.0, 1.0, 0.9, 0.1)
            p_noise  = p3c.slider("P(noise)",         0.0, 1.0, 0.5, 0.1)
            p_flip   = p4c.slider("P(flip)",          0.0, 1.0, 0.5, 0.1)
            n_aug    = st.slider("Samples to show", 2, 8, 4)

            gray = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2GRAY)
            _, fg = cv2.threshold(gray, 50, 1, cv2.THRESH_BINARY)
            fg = fg.astype(np.int64)
            tr = DomainRandomTransform(p_bg_replace=p_bg, p_color_jitter=p_jitter,
                                       p_noise=p_noise, p_flip=p_flip)
            augs = [tr(src_rgb.copy(), fg.copy())[0] for _ in range(n_aug)]
            show_cols = st.columns(n_aug + 1)
            show_cols[0].image(src_rgb, caption="Original", use_container_width=True)
            for i, aug in enumerate(augs):
                show_cols[i + 1].image(aug, caption=f"Aug #{i+1}", use_container_width=True)
        else:
            st.info("Choose or capture a source image above.")

# ===========================================================================
# Page 5 — 3D Object Scan
# ===========================================================================

elif page == "📦 3D Object Scan":
    st.title("📦 3D Object Scan — Reconstruct & Grasp Plan")

    # ── Camera angle — always visible, pre-populated from capture session ────
    _default_theta = int(round(st.session_state.get("cap_theta_deg", 90.0)))
    tc1, tc2 = st.columns([3, 1])
    with tc1:
        theta_deg = st.slider(
            "📐 Camera polar angle θ — degrees from zenith (set this to match how you captured)",
            min_value=0, max_value=90, value=_default_theta,
            help=(
                "**90°** = camera at object height, looking horizontally (turntable rig).\n\n"
                "**36° (π/5 rad)** = HEAPGrasp hand-eye default — arm above object, "
                "looking down at ~54° below horizontal.\n\n"
                "**0°** = camera directly overhead.\n\n"
                "This value is saved automatically when you use the Live Capture page."
            ),
        )
    with tc2:
        st.metric("θ in radians", f"{np.radians(theta_deg):.3f}")
        st.caption(f"{90 - theta_deg}° above horizontal")
    theta_rad = np.radians(theta_deg)

    # ── Other reconstruction parameters ──────────────────────────────────
    with st.expander("⚙️ More reconstruction parameters", expanded=False):
        rp1, rp2, rp3, rp4 = st.columns(4)
        V        = rp1.select_slider("Grid V³", [16, 24, 32, 48, 64], value=32)
        diameter = rp2.slider("Object diameter (m)", 0.05, 0.50, 0.15, 0.01)
        cam_dist = rp3.slider("Camera distance (m)", 0.20, 1.00, 0.40, 0.05)
        fov      = rp4.slider("Camera FOV (°)", 40.0, 100.0, 62.2, 0.5)

    # ── Input source selection ─────────────────────────────────────────
    n_captured = len(st.session_state.cap_frames)
    has_session = (n_captured >= 4 and st.session_state.cap_bg is not None)

    input_options = [
        "🔵 Synthetic sphere", "🏺 Synthetic mug",
        "🧱 Synthetic brick", "📐 Synthetic L-prism", "🍩 Synthetic torus",
        "📁 Upload masks",
    ]
    if has_session:
        input_options.insert(0, f"📷 Live session  ({n_captured} views captured)")

    input_mode = st.radio("Input source", input_options, horizontal=True)

    # ── Build silhouettes ─────────────────────────────────────────────
    if "Live session" in input_mode:
        bg = st.session_state.cap_bg

        with st.expander("✨ Contrast & shadow removal (live session)", expanded=False):
            lv_con, lv_shad = st.columns(2)
            with lv_con:
                st.markdown("**Contrast (CLAHE)**")
                live_contrast_on   = st.toggle("Enable contrast boost", value=False, key="live_contrast_on")
                live_clahe_clip    = st.slider("CLAHE clip limit", 1.0, 8.0, 2.0, 0.5,
                                               key="live_clahe_clip",
                                               help="Higher = stronger local contrast boost.")
            with lv_shad:
                st.markdown("**Shadow removal**")
                live_shadow_on     = st.toggle("Enable shadow removal", value=True, key="live_shadow_on")
                ls1, ls2, ls3      = st.columns(3)
                live_bg_thresh     = ls1.slider("BG threshold", 5, 80, 30, 5, key="live_bg_thresh")
                live_sh_dark       = ls2.slider("Darkness", 0.05, 0.50, 0.20, 0.05,
                                                key="live_sh_dark",
                                                help="Min L* drop to classify as shadow.")
                live_sh_chroma     = ls3.slider("Chroma tol.", 2.0, 30.0, 14.0, 1.0,
                                                key="live_sh_chroma",
                                                help="Max |Δa*|/|Δb*| for a shadow pixel.")

        raw_masks, angles = [], []
        for frame, ang in zip(st.session_state.cap_frames, st.session_state.cap_angles):
            if frame.shape != bg.shape:
                frame = np.array(
                    Image.fromarray(frame).resize((bg.shape[1], bg.shape[0]), Image.LANCZOS)
                )
            sess  = CaptureSession(frames=[frame], angles_deg=[ang], background=bg)
            mask  = extract_silhouettes(
                sess, method="background", bg_threshold=live_bg_thresh,
                shadow_removal=live_shadow_on,
                shadow_darkness=live_sh_dark,
                shadow_chroma_tol=live_sh_chroma,
                contrast_enhance=live_contrast_on,
                clahe_clip=live_clahe_clip,
            )[0]
            raw_masks.append(mask)
            angles.append(ang)

        st.success(f"Using live capture session: {len(raw_masks)} views.")
        strip_n = min(6, len(raw_masks))
        strip   = np.concatenate(
            [raw_masks[i].astype(np.uint8) * 255 for i in range(strip_n)], axis=1
        )
        st.image(strip, caption=f"First {strip_n} silhouettes", use_container_width=True)

    elif input_mode == "📁 Upload masks":
        files = st.file_uploader(
            "Silhouette masks (white=object) — one per view in rotation order",
            type=["png", "jpg", "bmp"], accept_multiple_files=True,
        )
        if not files:
            st.info("Upload at least 4 mask images.")
            st.stop()
        angles    = [i * 360.0 / len(files) for i in range(len(files))]
        raw_masks = [np.array(Image.open(f).convert("L")) > 127 for f in files]

    else:
        n_views_syn = st.slider("Synthetic views", 4, 24, 8)
        angles      = [i * 360.0 / n_views_syn for i in range(n_views_syn)]
        H, W        = 240, 320
        cy, cx      = H // 2, W // 2
        # Focal length consistent with cam_dist / diameter sliders
        fx_syn = (W / 2.0) / np.tan(np.radians(fov / 2.0))
        r_m    = diameter / 2.0
        r      = max(10, int(round(r_m * fx_syn / cam_dist)))

        if "mug" in input_mode:
            bw = max(8, int(r * 1.2))
            bh = max(10, int(r * 1.7))
            raw_masks = []
            for _ in angles:
                m = np.zeros((H, W), dtype=bool)
                m[cy-bh//2:cy+bh//2, cx-bw//2:cx+bw//2] = True
                yg, xg = np.ogrid[:H, :W]
                hcx, hcy = cx + bw//2 + int(r*0.28), cy + int(r*0.12)
                handle_a, handle_b = int(r*0.53), int(r*0.30)
                m[((yg-hcy)/handle_a)**2 + ((xg-hcx)/handle_b)**2 <= 1] = True
                m[((yg-hcy)/(handle_a*0.63))**2 + ((xg-hcx)/(handle_b*0.4))**2 <= 1] = False
                raw_masks.append(m)

        elif "brick" in input_mode:
            st.caption(
                "🧱 **Brick** — 2:1 aspect ratio in XZ. Silhouette is widest at 0°/180° "
                "and narrowest at 90°/270°. Reconstruction should be a flat slab, not a cube."
            )
            raw_masks = _make_shape_masks("brick", angles, H, W, cx, cy, fx_syn, cam_dist, r_m, theta_rad)

        elif "L-prism" in input_mode:
            st.caption(
                "📐 **L-prism** — L-shaped cross-section. Each view looks different; "
                "two arms of unequal length give a clearly asymmetric reconstruction."
            )
            raw_masks = _make_shape_masks("L", angles, H, W, cx, cy, fx_syn, cam_dist, r_m, theta_rad)

        elif "torus" in input_mode:
            st.caption(
                "🍩 **Torus** — ring standing upright. At 0° the silhouette is an annulus; "
                "at 90° it collapses to a thin rectangle. The most demanding SfS test."
            )
            raw_masks = _make_shape_masks("torus", angles, H, W, cx, cy, fx_syn, cam_dist, r_m, theta_rad)

        else:  # sphere
            raw_masks = []
            for _ in angles:
                m  = np.zeros((H, W), dtype=bool)
                yg, xg = np.ogrid[:H, :W]
                m[(yg-cy)**2 + (xg-cx)**2 <= r**2] = True
                raw_masks.append(m)

        strip_n = min(6, n_views_syn)
        strip   = np.concatenate(
            [raw_masks[i].astype(np.uint8) * 255 for i in range(strip_n)], axis=1
        )
        st.image(strip, caption=f"First {strip_n} silhouettes (note how width changes)",
                 use_container_width=True)

    # ── Run SfS ───────────────────────────────────────────────────────
    masks_bytes = tuple(m.astype(np.uint8).tobytes() for m in raw_masks)
    shapes_arg  = tuple(m.shape for m in raw_masks)
    mask_h, mask_w = raw_masks[0].shape
    K = default_camera_matrix(mask_w, mask_h, fov_deg=fov)
    with st.spinner(f"Carving {V}³ voxel grid from {len(raw_masks)} views…"):
        voxels, grid_center = _run_sfs(masks_bytes, shapes_arg, tuple(angles),
                                       V, diameter, cam_dist, fov, theta_rad)

    n_occ  = int(voxels.sum())
    fill   = 100.0 * n_occ / V**3
    pts_all = voxels_to_pointcloud(voxels, diameter, grid_center)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Occupied voxels", f"{n_occ:,}")
    mc2.metric("Fill ratio",      f"{fill:.1f}%")
    mc3.metric("Views used",      str(len(raw_masks)))
    if len(pts_all):
        ext = pts_all.max(0) - pts_all.min(0)
        mc4.metric("Bounding box",
                   f"{ext[0]*100:.1f}×{ext[1]*100:.1f}×{ext[2]*100:.1f} cm")

    st.markdown("---")

    # ── Reprojection Diagnostic ───────────────────────────────────────
    with st.expander("🔍 Reprojection Diagnostic", expanded=True):
        st.markdown(
            "Each tile shows one input silhouette overlaid with the reprojected hull. "
            "**Green** = hull only (over-carved / mask too small). "
            "**Blue** = mask only (under-carved). "
            "**White** = agreement. "
            "High IoU means the reconstruction is geometrically consistent."
        )

        thetas_list = [np.radians(theta_deg)] * len(angles)
        with st.spinner("Reprojecting hull…"):
            reproj_masks = reproject_hull(
                voxels, angles,
                camera_matrix=K,
                object_diameter=diameter,
                camera_distance=cam_dist,
                image_shape=raw_masks[0].shape,
                thetas_rad=thetas_list,
                grid_center=grid_center,
            )

        ious = []
        overlays = []
        for orig, proj in zip(raw_masks, reproj_masks):
            inter = (orig & proj).sum()
            union = (orig | proj).sum()
            ious.append(float(inter) / float(union) if union > 0 else 1.0)

            H_m, W_m = orig.shape
            rgb = np.zeros((H_m, W_m, 3), dtype=np.uint8)
            both  = orig & proj
            green = proj & ~orig          # hull only
            blue  = orig & ~proj          # mask only
            rgb[both,  :] = [255, 255, 255]
            rgb[green, :] = [80,  220,  80]
            rgb[blue,  :] = [80,  130, 255]
            overlays.append(rgb)

        mean_iou = float(np.mean(ious))
        ri1, ri2, ri3 = st.columns(3)
        ri1.metric("Mean IoU", f"{mean_iou:.3f}")
        ri2.metric("Best view IoU",  f"{max(ious):.3f}")
        ri3.metric("Worst view IoU", f"{min(ious):.3f}")

        if mean_iou >= 0.80:
            st.success("Good reprojection (IoU ≥ 0.80) — masks and camera model are consistent.")
        elif mean_iou >= 0.55:
            st.warning(
                "Moderate reprojection (IoU 0.55–0.80). "
                "Check camera distance, polar angle θ, and mask quality."
            )
        else:
            st.error(
                "Poor reprojection (IoU < 0.55). Likely causes: wrong camera distance or FOV, "
                "incorrect θ, or masks contain large shadows/noise."
            )

        max_cols = 4
        n_show   = min(len(overlays), 8)
        for row_start in range(0, n_show, max_cols):
            cols = st.columns(min(max_cols, n_show - row_start))
            for col, idx in zip(cols, range(row_start, row_start + len(cols))):
                col.image(
                    overlays[idx],
                    caption=f"View {idx+1}  IoU={ious[idx]:.2f}",
                    use_container_width=True,
                )

    st.markdown("---")

    # ── NBV Planner ────────────────────────────────────────────────────
    with st.expander("🎯 Next-Best-View Planner", expanded=True):
        nbv_c1, nbv_c2 = st.columns([2, 1])
        n_nbv    = nbv_c1.slider("Suggested additional views", 1, 8, 3)
        lam      = nbv_c2.slider("λ (trajectory penalty)", 0.0, 2.0, 0.5, 0.1)
        run_nbv  = st.toggle("Show in 3D view", value=True)

        planner = NextBestViewPlanner(
            camera_matrix=K, volume_size=V,
            object_diameter=diameter, camera_distance=cam_dist, lam=lam,
        )
        for ang in angles:
            planner.update(ang, voxels)

        nbv_suggestions, nbv_igs = [], []
        for _ in range(n_nbv):
            nbv = planner.next_best_view()
            nbv_suggestions.append(nbv)
            nbv_igs.append(planner._information_gain(nbv))
            planner.update(nbv, voxels)

        for step_i, (ang, ig) in enumerate(zip(nbv_suggestions, nbv_igs), 1):
            st.markdown(f"- **Step {step_i}:** rotate to **{ang:.1f}°** &nbsp; (IG = {ig:.0f})")

    st.markdown("---")

    # ── Antipodal Grasp Planning ──────────────────────────────────────
    st.subheader("🤏 Antipodal Grasp Candidates")
    gc1, gc2 = st.columns([2, 1])
    max_jaw  = gc1.slider("Max jaw opening (mm)", 20, 120, 80, 5) / 1000.0
    n_grasps = gc2.number_input("Top-K candidates", 1, 8, 3)

    with st.spinner("Computing grasp candidates…"):
        grasps = find_grasps(
            voxels, diameter, grid_center,
            max_jaw_width=max_jaw,
            n_directions=512,
            top_k=int(n_grasps),
        )

    if not grasps:
        st.warning("No valid grasp found — try increasing max jaw opening or capturing more views.")
    else:
        grasp_colors = ["🟢", "🟡", "🟠", "🟣", "🔵"]
        for rank, g in enumerate(grasps):
            col_icon = grasp_colors[rank % len(grasp_colors)]
            with st.expander(
                f"{col_icon} Grasp #{rank+1}  —  score {g.score:.3f}  "
                f"|  width {g.width*1000:.1f} mm",
                expanded=(rank == 0),
            ):
                ga, gb, gc_ = st.columns(3)
                ga.metric("Overall score",   f"{g.score:.3f}")
                gb.metric("Normal (friction)",f"{g.normal_score:.3f}")
                gc_.metric("Width score",     f"{g.width_score:.3f}")

                gd, ge, gf = st.columns(3)
                gd.metric("Jaw opening",  f"{g.width*1000:.1f} mm")
                ge.metric("Center score", f"{g.center_score:.3f}")
                gf.metric("Rank", f"#{rank+1} / {len(grasps)}")

                c1c, c2c, appc = st.columns(3)
                c1c.markdown(
                    f"**Jaw A contact**  \n"
                    f"X={g.contact_1[0]*100:.1f} cm  \n"
                    f"Y={g.contact_1[1]*100:.1f} cm  \n"
                    f"Z={g.contact_1[2]*100:.1f} cm"
                )
                c2c.markdown(
                    f"**Jaw B contact**  \n"
                    f"X={g.contact_2[0]*100:.1f} cm  \n"
                    f"Y={g.contact_2[1]*100:.1f} cm  \n"
                    f"Z={g.contact_2[2]*100:.1f} cm"
                )
                appc.markdown(
                    f"**Approach dir**  \n"
                    f"X={g.approach_dir[0]:.2f}  \n"
                    f"Y={g.approach_dir[1]:.2f}  \n"
                    f"Z={g.approach_dir[2]:.2f}"
                )

    st.markdown("---")

    # ── 3D interactive figure ─────────────────────────────────────────
    st.subheader("Interactive 3D View")
    show_grasp = st.checkbox("Show grasp candidates", value=True)

    try:
        fig = _make_scan_figure(
            voxels=voxels, diameter=diameter, cam_dist=cam_dist,
            visited_angles=angles,
            nbv_angles=nbv_suggestions if run_nbv else None,
            show_grasp=show_grasp,
            grid_center=grid_center,
            grasps=grasps if show_grasp else None,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("Install plotly: `pip install plotly`")

    st.caption("🟣 Point cloud  ·  🔴/🟢/🟡 Grasp candidates (line = jaw span, arrow = approach)  "
               "·  🟦 Captured views  ·  🟠 NBV suggestions")

    # ── Cross-section slices ──────────────────────────────────────────
    with st.expander("Cross-section slices"):
        slice_idx = st.slider("Slice index", 0, V - 1, V // 2)
        fig2, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#0e1117")
        for ax, (data, title) in zip(axes, [
            (voxels[:, :, slice_idx].T, f"XY  (Z={slice_idx})"),
            (voxels[:, slice_idx, :].T, f"XZ  (Y={slice_idx})"),
            (voxels[slice_idx, :, :].T, f"YZ  (X={slice_idx})"),
        ]):
            ax.imshow(data, origin="lower", cmap="plasma", vmin=0, vmax=1)
            ax.set_title(title, color="white")
            ax.set_facecolor("#1e1e2e")
            ax.tick_params(colors="gray")
            for sp in ax.spines.values():
                sp.set_edgecolor("#313244")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Download PLY ──────────────────────────────────────────────────
    st.download_button(
        "⬇️ Download point cloud (.ply)",
        data=_ply_bytes(voxels, diameter, grid_center),
        file_name="heapgrasp_reconstruction.ply",
        mime="application/octet-stream",
    )

    # ── Grasp Plan (text summary of best grasp) ───────────────────────
    st.markdown("---")
    st.subheader("📋 Grasp Execution Plan — Ford Industrial Arm")
    plan_text = generate_grasp_plan(voxels, diameter, cam_dist, len(raw_masks),
                                    grid_center=grid_center)
    st.markdown(f"<div class='grasp-plan'>{plan_text}</div>", unsafe_allow_html=True)

    copy_col, _ = st.columns([1, 3])
    copy_col.download_button(
        "⬇️ Download grasp plan (.txt)",
        data=plan_text.encode(),
        file_name="grasp_plan.txt",
        mime="text/plain",
    )

# ===========================================================================
# Page 6 — Hailo NPU
# ===========================================================================

elif page == "⚡ Hailo NPU":
    st.title("⚡ Hailo-8L AI Hat+ — NPU Status")

    if _HAILO_OK:
        st.success(
            f"✅ **Hailo-8L detected** — HailoRT {_HAILO_VERSION}  |  "
            "13 TOPS · ~4 W · PCIe M.2 on Pi 5"
        )
    else:
        st.error(
            "❌ **Hailo-8L not detected.**  Make sure the AI Hat+ is seated on the "
            "40-pin GPIO and HailoRT is installed.  "
            "See the Hailo Developer Zone for install instructions."
        )

    st.markdown("---")
    st.subheader("Estimated inference latency: CPU vs NPU")

    import plotly.graph_objects as go
    rows = [
        ("MobileNetV3-S classify",  10,    2),
        ("DeepLabv3+ segment",      2000,  60),
        ("SlipDetectorCNN tactile", 5,     0.8),
        ("SfS 64³ reconstruct",     400,   400),
    ]
    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(name="Pi 5 CPU", x=[r[0] for r in rows],
                           y=[r[1] for r in rows], marker_color="#60a5fa",
                           text=[f"{r[1]} ms" for r in rows], textposition="outside"))
    fig_p.add_trace(go.Bar(name="Hailo-8L NPU", x=[r[0] for r in rows],
                           y=[r[2] for r in rows], marker_color="#a6e3a1",
                           text=[f"{r[2]} ms" for r in rows], textposition="outside"))
    fig_p.update_layout(
        barmode="group",
        yaxis=dict(title="Latency (ms)", type="log", gridcolor="#313244"),
        xaxis_gridcolor="#313244",
        legend=dict(bgcolor="rgba(30,30,46,0.8)", bordercolor="#45475a"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=380, margin=dict(t=20, b=0),
    )
    st.plotly_chart(fig_p, use_container_width=True)
    st.caption("NPU figures are estimated INT8 targets. SfS is pure numpy — identical on both.")

    st.markdown("---")
    st.subheader("Compile workflow")

    wf_cols = st.columns(4)
    for col, (step, cmd, color) in zip(wf_cols, [
        ("1️⃣ Train",       "gripper-cv-train-seg",         "#3b82f6"),
        ("2️⃣ Export ONNX", "gripper-cv-export-onnx",       "#8b5cf6"),
        ("3️⃣ Compile HEF", "hailo optimize\n+ hailo compile", "#ec4899"),
        ("4️⃣ Run on NPU",  "--segment hailo\n--hef-path …",  "#a6e3a1"),
    ]):
        with col:
            st.markdown(
                f"<div style='border:1px solid {color};border-radius:8px;"
                f"padding:12px;text-align:center'>"
                f"<b>{step}</b><br><code style='font-size:.75em'>{cmd}</code></div>",
                unsafe_allow_html=True)

    st.markdown("")
    st.code("""# 1. Train
gripper-cv-train-seg --synthetic-length 2000 --epochs 20 --output-dir outputs/seg

# 2. Export to ONNX
gripper-cv-export-onnx --checkpoint outputs/seg/best_model.pt --output model.onnx

# 3. Compile to HEF  (machine with Hailo Dataflow Compiler)
hailo optimize model.onnx --hw-arch hailo8l --calib-path /calib
hailo compile  model.har  --hw-arch hailo8l --output-dir .

# 4. Run on the Pi with NPU
gripper-cv-heapgrasp --segment hailo --hef-path model.hef""", language="bash")

    st.markdown("---")
    st.subheader("Python API")
    st.code("""from gripper_cv.hailo import HailoRunner, is_hailo_available

print(is_hailo_available())          # True on Pi with AI Hat+

# Generic inference
with HailoRunner("model.hef") as runner:
    out = runner.run_single(input_nchw_float32)  # returns NCHW

# Segmentation on NPU
from gripper_cv.heapgrasp.segment import extract_silhouettes
masks = extract_silhouettes(session, method="hailo", hef_path="seg.hef")

# Tactile slip detection on NPU
from gripper_cv.sim2real.tactile import SlipDetector
det = SlipDetector(hef_path="slip_cnn.hef")
is_slipping, prob = det.predict(tactile_frame_rgb)""", language="python")
