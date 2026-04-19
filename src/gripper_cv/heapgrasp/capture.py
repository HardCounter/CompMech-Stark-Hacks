"""
Interactive multi-view capture for HEAPGrasp Shape-from-Silhouette.

Turntable protocol:
  1. Remove object → SPACE → background captured
  2. Place object at 0° → SPACE → view 0
  3. Rotate by step_deg → SPACE → ... repeat N times
  4. Q to quit

Frames are stored as RGB arrays. Background frame enables fast
background-subtraction segmentation without a heavy CNN.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from gripper_cv.camera import CameraConfig, PiCameraStream


@dataclass
class CaptureSession:
    frames: List[np.ndarray] = field(default_factory=list)      # RGB, per view
    angles_deg: List[float] = field(default_factory=list)
    background: Optional[np.ndarray] = None                      # RGB background


def capture_multiview(
    n_views: int = 8,
    output_dir: Optional[Path] = None,
    width: int = 640,
    height: int = 480,
) -> CaptureSession:
    """
    Capture N silhouette views interactively.

    User removes object and presses SPACE for background, then places the object
    and presses SPACE once per view after rotating it by 360/n_views degrees.
    """
    session = CaptureSession()
    step = 360.0 / n_views

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    cfg = CameraConfig(width=width, height=height, fps=30)
    cam = PiCameraStream(cfg)
    cam.start()
    time.sleep(1.0)

    # -1 = waiting for background; 0..n_views-1 = object views
    state = -1

    def _overlay(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        return out

    print("HEAPGrasp capture started. Press SPACE to capture, Q to quit.")

    try:
        while True:
            frame_rgb = cam.read_rgb()
            display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if state == -1:
                label = "REMOVE object. Place ArUco marker flat on turntable. Press SPACE."
            elif state < n_views:
                label = f"View {state+1}/{n_views} @ {state*step:.0f}deg — Press SPACE"
            else:
                label = "All views done. Press Q to finish."

            cv2.imshow("HEAPGrasp Capture", _overlay(display, label))
            key = cv2.waitKey(30) & 0xFF

            if key == ord(" "):
                # Grab a frame that arrived *after* SPACE was pressed so we
                # don't accidentally save a frame from before the user acted.
                captured = cam.wait_for_fresh_frame()
                if state == -1:
                    session.background = captured
                    print("Background captured. Place object at 0° and press SPACE.")
                    state = 0
                elif state < n_views:
                    angle = state * step
                    session.frames.append(captured)
                    session.angles_deg.append(angle)
                    print(f"  View {state+1}/{n_views} captured at {angle:.0f}°")
                    if output_dir:
                        bgr = cv2.cvtColor(captured, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(output_dir / f"view_{state:03d}.jpg"), bgr)
                    state += 1
                    if state < n_views:
                        print(f"  Rotate object to {state*step:.0f}° and press SPACE.")
                    else:
                        print("All views captured. Press Q to finish.")

            elif key in (ord("q"), ord("Q")):
                break

    finally:
        cv2.destroyAllWindows()
        cam.close()

    print(f"Session complete: {len(session.frames)} views, background={'yes' if session.background is not None else 'no'}")
    return session
