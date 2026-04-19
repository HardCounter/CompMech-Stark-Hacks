from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    device: int = 0           # V4L2 device index (/dev/video<device>)
    square_crop: bool = False # crop to square (min(w,h)) before returning — useful for CNN inputs


class PiCameraStream:
    """
    USB webcam wrapper with the same interface as the old picamera2-based class.

    Uses a background thread to continuously drain the V4L2 buffer so that
    read_bgr() / read_rgb() always return the freshest available frame rather
    than a stale buffered one. This matters most during interactive capture
    (heapgrasp turntable) where the user positions the object and then presses
    SPACE — they need the frame that was live at that moment.
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        # Force V4L2 backend — avoids GStreamer pipeline failures on Raspberry Pi OS.
        self._cap = cv2.VideoCapture(self.config.device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            print(
                f"No USB camera detected at /dev/video{self.config.device}. "
                "Check that the device is connected and not held by another process "
                "(e.g. ls /dev/video*).",
                file=sys.stderr,
            )
            raise RuntimeError(f"Cannot open /dev/video{self.config.device}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        # Limit V4L2 internal buffer to 1 frame so read() always returns the
        # freshest frame rather than one that was queued seconds ago.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Try to lock exposure so background and object frames are comparable.
        # V4L2 exposure mode: 1 = manual, 3 = aperture-priority/auto.
        # This is best-effort — not all webcams honour it.
        if not self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1):
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # some cameras use 0 for manual

        self._lock = threading.Lock()
        self._latest_bgr: Optional[np.ndarray] = None
        self._frame_count: int = 0          # monotonic; used by wait_for_fresh_frame()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        # Wait until at least 10 frames have arrived. USB webcams need several
        # frames for auto-exposure to settle after open; accepting the very
        # first frame as "ready" risks capturing a dark initialisation frame.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame_count >= 10:
                    break
            time.sleep(0.02)
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._running = False

    def close(self) -> None:
        self.stop()
        self._cap.release()

    def __enter__(self) -> "PiCameraStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read_bgr(self) -> np.ndarray:
        if not self._running:
            raise RuntimeError("Camera is not started. Call start() first.")
        with self._lock:
            frame = self._latest_bgr
        if frame is None:
            raise RuntimeError("No frame available yet.")
        return self._maybe_crop(frame)

    def read_rgb(self) -> np.ndarray:
        return cv2.cvtColor(self.read_bgr(), cv2.COLOR_BGR2RGB)

    def wait_for_fresh_frame(self, timeout: float = 1.0) -> np.ndarray:
        """
        Block until a frame arrives that is strictly newer than the one
        currently in the buffer, then return it as RGB.

        Call this immediately before capturing a turntable view so that the
        saved frame reflects the scene at the moment of the SPACE press, not
        the frame that happened to be buffered from 30 ms earlier.
        """
        if not self._running:
            raise RuntimeError("Camera is not started.")
        with self._lock:
            seen = self._frame_count
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame_count > seen:
                    frame = self._latest_bgr
                    break
            time.sleep(0.005)
        else:
            frame = self._latest_bgr  # timeout fallback
        return cv2.cvtColor(self._maybe_crop(frame), cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                print("USB camera: failed to read frame.", file=sys.stderr)
                self._stop.set()
                break
            with self._lock:
                self._latest_bgr = frame
                self._frame_count += 1

    def _maybe_crop(self, frame: np.ndarray) -> np.ndarray:
        if not self.config.square_crop:
            return frame
        h, w = frame.shape[:2]
        if w == h:
            return frame
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        return frame[y0:y0 + side, x0:x0 + side]
