  Stage 1b — Auto-calibration (calibrate.py)

  Camera distance via ArUco:

  detect_aruco_marker() converts the background frame to grayscale, runs
  cv2.aruco.ArucoDetector (DICT_4X4_50 — 4×4 binary patterns, 50 distinct IDs). When a marker
  is found, it calls:

  cv2.solvePnP(obj_pts, img_pts, K, None, flags=SOLVEPNP_IPPE_SQUARE)

  obj_pts are the four corners of the marker in its own local frame (a flat square centred at
  origin, Z=0). SOLVEPNP_IPPE_SQUARE is the exact solver for this symmetric planar case — it's
   algebraic (not iterative), so it always converges in one shot. The result is [rvec, tvec]
  where tvec is the marker's centre in camera coordinates (metres). Since the marker is on the
   turntable centre:

  camera_distance = ||tvec||₂ = sqrt(tx² + ty² + tz²)

  Object diameter via silhouette projection:

  After segmentation, estimate_object_diameter() inverts the pinhole equation. For each mask's
   bounding box (w_px, h_px) at depth Z = camera_distance:

  W_m = w_px * Z / fx
  H_m = h_px * Z / fy

  Take the max across all views (the widest cross-section = true diameter) then multiply by
  1.05 as a safety margin. This ensures the voxel grid always fully contains the object.

  ---
  Stage 2 — Silhouette extraction (segment.py)

  Background subtraction (default):

  diff = cv2.absdiff(frame_rgb, background_rgb).mean(axis=2)
  mask = diff > threshold

  Per-pixel absolute difference, averaged across RGB channels. Anything that changed more than
   threshold (default 30 out of 255) is foreground. Followed by morphological close (fills
  small holes) then open (removes noise blobs) with a 9×9 elliptical kernel. Result: clean
  binary mask (H, W) bool.

  DeepLabv3+ (optional):

  Loads deeplabv3_resnet50 pretrained on PASCAL VOC 21 classes. Runs inference on each frame,
  takes argmax of the 21-class output — any non-zero class (i.e., not background) becomes the
  mask. No fine-tuning, just ImageNet/VOC weights. Works okay for general objects, worse for
  metallic/transparent automotive parts.

  ---
  Stage 3 — Shape from Silhouette (reconstruct.py:45)

  This is the core algorithm. The central idea: if a 3D point is not inside the object, there
  must be at least one view from which it would be visible but NOT within the silhouette. By
  carving all such points, what remains is the visual hull — the intersection of all
  silhouette cones.

  Setup:

  A 3D voxel grid of V³ voxels (default 64³ = 262,144 voxels) is initialized as all-occupied.
  The grid spans [-object_diameter/2, +object_diameter/2] in all three axes, centred at the
  world origin (= turntable centre).

  lin = np.linspace(-half, half, V)   # V equally-spaced values
  xi, yi, zi = np.meshgrid(lin, lin, lin, indexing='ij')
  pts = np.stack([xi.ravel(), yi.ravel(), zi.ravel()], axis=0)  # shape: (3, V³)

  All V³ points are processed in parallel as one matrix — no Python loop over individual
  voxels.

  For each view:

  The turntable model treats the camera as fixed at (0, 0, camera_distance) in world space,
  with the object rotating. Rotating the object by angle_deg is equivalent to rotating the
  camera by -angle_deg around the world Y axis:

  theta = -angle_deg  # radians
  Ry = [[cos θ,  0, sin θ],
        [0,      1, 0    ],
        [-sin θ, 0, cos θ]]

  Transform world points to camera frame:
  pts_cam = Ry @ pts          # rotate
  pts_cam[2] += camera_distance   # translate camera to (0,0,d)

  Perspective project all V³ points at once:
  proj = K @ pts_cam           # (3, V³)
  u = proj[0] / depth
  v = proj[1] / depth

  For each voxel, look up whether its projection (u, v) lands inside the silhouette:
  in_sil[in_frame] = mask[vi[in_frame], ui[in_frame]]

  Carve any voxel that projects inside the image frame but outside the silhouette:
  occupied &= ~(in_frame & ~in_sil)

  Voxels outside the image frustum are left alone (conservative — we have no information from
  that view).

  Result: occupied.reshape(V, V, V) — a 3D boolean volume. True = inside the visual hull.

  What the visual hull is and isn't:

  - It's a guaranteed upper bound — the real object is a subset of the hull
  - Concave regions (like the inside of a cup, or a hole through a part) cannot be
  reconstructed — silhouettes only see the outer envelope
  - Accuracy improves with more views and better segmentation
  - At 64³ with a 15cm object, each voxel is 15/64 ≈ 2.3mm — reasonable for grasp planning

  ---
  Stage 4 — Export (export.py)

  voxels_to_pointcloud() converts occupied voxel indices back to metric coordinates via the
  same linspace, giving an (N, 3) float array in metres. Written to an ASCII .ply file (open
  in MeshLab, CloudCompare, or Open3D).
