from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError as exc:
    raise ImportError(
        "picamera2 is required. Install on Raspberry Pi OS with: "
        "sudo apt install -y python3-picamera2"
    ) from exc


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30


class PiCameraStream:
    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._camera = Picamera2()
        video_cfg = self._camera.create_video_configuration(
            main={"size": (self.config.width, self.config.height)},
        )
        self._camera.configure(video_cfg)
        self._running = False

    def start(self) -> None:
        if not self._running:
            self._camera.start()
            self._running = True

    def read_bgr(self) -> np.ndarray:
        if not self._running:
            raise RuntimeError("Camera is not started. Call start() first.")
        frame = self._camera.capture_array("main")
        # capture_array may return XBGR (4-ch) or BGR (3-ch) depending on format
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def read_rgb(self) -> np.ndarray:
        return cv2.cvtColor(self.read_bgr(), cv2.COLOR_BGR2RGB)

    def stop(self) -> None:
        if self._running:
            self._camera.stop()
            self._running = False

    def close(self) -> None:
        self.stop()
        self._camera.close()

    def __enter__(self) -> "PiCameraStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()
