"""
HEAPGrasp end-to-end pipeline:
  capture → auto-calibrate → segment → estimate size → reconstruct (SfS) → export
"""

from __future__ import annotations

from pathlib import Path

from .calibrate import CalibrationResult, auto_calibrate, estimate_object_diameter
from .capture import capture_multiview
from .export import save_masks, save_ply, save_voxels_npy
from .reconstruct import default_camera_matrix, shape_from_silhouette, voxels_to_pointcloud
from .segment import extract_silhouettes


def run_pipeline(
    n_views: int = 8,
    segment_method: str = "background",
    volume_size: int = 64,
    object_diameter: float = 0.15,
    camera_distance: float = 0.40,
    fov_deg: float = 62.2,
    bg_threshold: int = 30,
    output_dir: str = "outputs/heapgrasp",
    width: int = 640,
    height: int = 480,
    device: str = "cpu",
    auto_scale: bool = True,
    marker_size_m: float = 0.05,
    seg_checkpoint: str | None = None,
    use_planner: bool = False,
    n_planner_views: int = 4,
    hef_path: str | None = None,
    seg_img_size: tuple = (512, 512),
    theta_deg: float = 90.0,
    thetas_deg: list | None = None,
) -> None:
    """
    Run the full HEAPGrasp pipeline.

    Args:
        n_views:          number of views (evenly spaced azimuthally, 360/n_views deg apart)
        segment_method:   "background", "deeplab", "finetuned", or "hailo"
        volume_size:      voxel grid resolution (volume_size³ voxels)
        object_diameter:  manual reconstruction volume side length in metres
        camera_distance:  hemisphere radius in metres
        fov_deg:          camera horizontal FOV (Pi Cam Module 3 ≈ 66°, v2 ≈ 62.2°)
        bg_threshold:     background subtraction sensitivity (lower = more sensitive)
        output_dir:       directory for all outputs
        width, height:    capture resolution
        device:           torch device for deeplab/finetuned ("cpu", "cuda", "mps")
        auto_scale:       if True, auto-detect distances via ArUco and silhouettes
        marker_size_m:    physical ArUco marker side length in metres
        seg_checkpoint:   path to fine-tuned .pt; switches method to "finetuned" automatically
        use_planner:      if True, run the NBV planner after reconstruction
        n_planner_views:  number of additional view angles the planner suggests
        hef_path:         path to compiled .hef for Hailo-8L NPU (method="hailo")
        seg_img_size:     (H, W) the Hailo model was compiled for (hailo method only)
        theta_deg:        polar angle from zenith for all views (degrees).
                          90 = horizontal side view (legacy turntable).
                          HEAPGrasp hand-eye uses ~36° (π/5 rad).
        thetas_deg:       per-view polar angles (degrees); overrides theta_deg if given.
                          Use this when the robot arm provides exact viewpoint poses.
    """
    out = Path(output_dir)

    # Auto-promote segment method
    effective_method = segment_method
    if hef_path is not None and segment_method == "background":
        effective_method = "hailo"
        print("  [pipeline] hef_path given — switching to method='hailo' (Hailo-8L NPU)")
    elif seg_checkpoint is not None and segment_method == "background":
        effective_method = "finetuned"
        print("  [pipeline] seg_checkpoint given — switching to method='finetuned'")

    print("=" * 50)
    print("HEAPGrasp — Shape from Silhouette Pipeline")
    print(f"  Views: {n_views}  |  Segmentation: {effective_method}  |  Grid: {volume_size}³")
    print(f"  Auto-scale: {'on' if auto_scale else 'off (manual)'}")
    if use_planner:
        print(f"  NBV planner: ON — will suggest {n_planner_views} additional views")
    print("=" * 50)

    # K is needed for both calibration and reconstruction; build it first
    K = default_camera_matrix(width=width, height=height, fov_deg=fov_deg)

    # ── 1. Capture ────────────────────────────────────────────────────────
    session = capture_multiview(
        n_views=n_views,
        output_dir=out / "frames",
        width=width,
        height=height,
    )
    if not session.frames:
        print("No frames captured — exiting.")
        return

    # ── 1b. Auto-calibrate: camera distance from ArUco marker ─────────────
    cal: CalibrationResult | None = None
    if auto_scale:
        print("\nAuto-calibrating…")
        cal = auto_calibrate(
            session, K,
            marker_size_m=marker_size_m,
            fallback_diameter=object_diameter,
            fallback_distance=camera_distance,
        )
        camera_distance = cal.camera_distance

    # ── 2. Segment ────────────────────────────────────────────────────────
    print(f"\nExtracting silhouettes ({effective_method})…")
    masks = extract_silhouettes(
        session,
        method=effective_method,
        device=device,
        bg_threshold=bg_threshold,
        checkpoint=seg_checkpoint,
        hef_path=hef_path,
        img_size=seg_img_size,
    )
    save_masks(masks, out / "masks")

    # ── 2c. Mask quality check ────────────────────────────────────────────
    fill_ratios = [m.mean() for m in masks]
    for i, r in enumerate(fill_ratios):
        print(f"  Mask {i+1}: {r*100:.1f}% foreground")

    empty = [i for i, r in enumerate(fill_ratios) if r < 0.005]
    if len(empty) == len(masks):
        raise RuntimeError(
            "\nAll silhouette masks are empty (<0.5% foreground).\n"
            "Most likely cause: the BACKGROUND was captured WITH the object in place,\n"
            "so every view looks identical to the background → nothing detected.\n"
            "Fix: re-run and make sure the object is fully removed before pressing\n"
            "SPACE for the background frame.\n"
            "Check outputs/heapgrasp/masks/ to inspect the masks visually."
        )
    if len(empty) > len(masks) // 2:
        print(
            f"  WARNING: {len(empty)}/{len(masks)} masks are nearly empty — "
            "reconstruction quality will be poor.\n"
            "  Likely cause: lighting changed between background and object captures,\n"
            "  or the object blends with the background. Try a contrasting backdrop."
        )

    # ── 2b. Auto-calibrate: object diameter from silhouette bounding boxes ─
    if auto_scale:
        object_diameter = estimate_object_diameter(
            masks, K, camera_distance,
            fallback_diameter=object_diameter,
        )
        source = cal.source if cal else "silhouette_only"
        if cal and cal.source == "aruco":
            source = "aruco+silhouette"
        print(f"  [calibrate] Object diameter: {object_diameter:.3f} m  (source: {source})")

    print(f"\n  camera_distance = {camera_distance:.3f} m  |  object_diameter = {object_diameter:.3f} m")

    # ── 3. Reconstruct ────────────────────────────────────────────────────
    import math as _math
    n_v = len(session.angles_deg)
    if thetas_deg is not None:
        thetas_rad_list = [_math.radians(t) for t in thetas_deg]
    else:
        thetas_rad_list = [_math.radians(theta_deg)] * n_v

    print(f"\nShape from Silhouette ({volume_size}³ voxels)…")
    print(f"  Camera polar angle θ: {theta_deg:.1f}° from zenith")
    voxels = shape_from_silhouette(
        silhouettes=masks,
        angles_deg=session.angles_deg,
        camera_matrix=K,
        volume_size=volume_size,
        object_diameter=object_diameter,
        camera_distance=camera_distance,
        thetas_rad=thetas_rad_list,
    )

    n_occ = int(voxels.sum())
    total = voxels.size
    print(f"  Visual hull: {n_occ:,} / {total:,} voxels ({100*n_occ/total:.1f}%)")

    # ── 3b. Next-Best-View planner ────────────────────────────────────────
    if use_planner:
        from gripper_cv.planner import NextBestViewPlanner
        planner = NextBestViewPlanner(
            camera_matrix=K,
            volume_size=volume_size,
            object_diameter=object_diameter,
            camera_distance=camera_distance,
        )
        for ang in session.angles_deg:
            planner.update(ang, voxels)
        print(f"\nNext-Best-View planner — suggesting {n_planner_views} additional views:")
        suggested = []
        for step_i in range(n_planner_views):
            nbv = planner.next_best_view()
            ig = planner._information_gain(nbv)
            print(f"  Step {step_i+1}: θ = {nbv:.1f}°  (IG={ig:.1f})")
            suggested.append(nbv)
            planner.update(nbv, voxels)   # simulate the view so next pick is different
        print(
            f"\n  Rotate to these angles and re-run with n_views={n_views + n_planner_views} "
            "for a more accurate reconstruction."
        )

    # ── 4. Export ─────────────────────────────────────────────────────────
    points = voxels_to_pointcloud(voxels, object_diameter)
    save_ply(points, out / "reconstruction.ply")
    save_voxels_npy(voxels, out / "voxels.npy")

    print("\nDone! Open outputs/heapgrasp/reconstruction.ply in MeshLab or CloudCompare.")
