from __future__ import annotations

import argparse
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import simplejpeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MJPEG stream server from Pi camera.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--quality", type=int, default=75)
    return parser.parse_args()


def make_handler(cam, quality: int, interval_s: float):
    class MJPEGHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-type", "multipart/x-mixed-replace; boundary=jpgboundary")
            self.end_headers()
            try:
                while True:
                    frame_rgb = cam.read_rgb()
                    jpeg = simplejpeg.encode_jpeg(frame_rgb, quality=quality)
                    self.wfile.write(b"--jpgboundary\r\n")
                    self.send_header("Content-type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    time.sleep(interval_s)
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    return MJPEGHandler


def main() -> None:
    args = parse_args()
    from gripper_cv.camera import CameraConfig, PiCameraStream

    cfg = CameraConfig(width=args.width, height=args.height, fps=args.fps)
    interval_s = 1.0 / max(args.fps, 1)
    with PiCameraStream(cfg) as cam:
        handler = make_handler(cam, args.quality, interval_s)
        server = HTTPServer((args.host, args.port), handler)
        print(f"Stream running at http://<PI_IP>:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
