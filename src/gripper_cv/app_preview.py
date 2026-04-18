from __future__ import annotations

import argparse

from gripper_cv.utils import FpsMeter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raspberry Pi camera preview.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from gripper_cv.camera import CameraConfig, PiCameraStream

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV (cv2) is required for preview. "
            "Install on Raspberry Pi OS with: sudo apt install -y python3-opencv"
        ) from exc

    cfg = CameraConfig(width=args.width, height=args.height, fps=args.fps)
    fps_meter = FpsMeter()

    with PiCameraStream(cfg) as cam:
        while True:
            frame = cam.read_bgr()
            fps = fps_meter.tick()
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("gripper-cv-preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
