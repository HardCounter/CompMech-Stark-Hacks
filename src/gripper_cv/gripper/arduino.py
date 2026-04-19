"""
Serial driver for the HEAPGrasp MG996R servo gripper (Arduino Nano).

Protocol (115200 baud, CR+LF terminated ASCII):
  OPEN          → move to open_angle
  CLOSE         → move to close_angle
  ANGLE <deg>   → move to exact angle (0–180)
  STATUS        → reply current angle in degrees
  STOP          → detach servo (saves power, stops jitter)

All replies: "OK <payload>\\r\\n"  or  "ERR <reason>\\r\\n"

Jaw-width mapping
-----------------
grip(width_mm) linearly maps the planned jaw opening onto the servo range:

    angle = open_angle + (close_angle - open_angle) * (1 - width_mm / max_jaw_mm)

So width_mm=max_jaw_mm → open_angle, width_mm=0 → close_angle.
Tune open_angle / close_angle to match your gripper's physical limits.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import serial


class GripperError(RuntimeError):
    pass


class GripperController:
    """
    Thread-safe serial interface to the Arduino MG996R servo gripper.

    Usage::

        with GripperController("/dev/ttyUSB0") as g:
            g.open_jaws()
            g.grip(width_mm=40.0)
    """

    DEFAULT_PORT    = "/dev/ttyUSB0"
    DEFAULT_BAUD    = 115200
    DEFAULT_TIMEOUT = 5.0
    BOOT_DELAY      = 2.5   # wait for Arduino bootloader after DTR reset

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baud: int = DEFAULT_BAUD,
        open_angle: int = 30,      # degrees — jaws fully open (match sketch)
        close_angle: int = 150,    # degrees — jaws fully closed (match sketch)
        max_jaw_mm: float = 80.0,  # physical jaw span at open_angle
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._port        = port
        self._baud        = baud
        self._open_angle  = open_angle
        self._close_angle = close_angle
        self._max_jaw_mm  = max_jaw_mm
        self._timeout     = timeout
        self._ser: Optional[serial.Serial] = None
        self._lock        = threading.Lock()

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> "GripperController":
        self._ser = serial.Serial(
            port     = self._port,
            baudrate = self._baud,
            bytesize = serial.EIGHTBITS,
            parity   = serial.PARITY_NONE,
            stopbits = serial.STOPBITS_ONE,
            timeout  = self._timeout,
            xonxoff  = False,
            rtscts   = False,
            dsrdtr   = False,
        )
        self._ser.dtr = False
        time.sleep(self.BOOT_DELAY)
        self._ser.reset_input_buffer()
        return self

    def disconnect(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def __enter__(self) -> "GripperController":
        return self.connect()

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    @property
    def open_angle(self) -> int:
        return self._open_angle

    @open_angle.setter
    def open_angle(self, v: int) -> None:
        self._open_angle = int(max(0, min(180, v)))

    @property
    def close_angle(self) -> int:
        return self._close_angle

    @close_angle.setter
    def close_angle(self, v: int) -> None:
        self._close_angle = int(max(0, min(180, v)))

    @property
    def max_jaw_mm(self) -> float:
        return self._max_jaw_mm

    @max_jaw_mm.setter
    def max_jaw_mm(self, v: float) -> None:
        if v <= 0:
            raise ValueError("max_jaw_mm must be positive")
        self._max_jaw_mm = v

    # ── Low-level ─────────────────────────────────────────────────────────────

    def _send(self, command: str) -> str:
        if not self.is_connected:
            raise GripperError("Gripper not connected — call connect() first")
        with self._lock:
            self._ser.reset_input_buffer()
            self._ser.write((command.strip() + "\r\n").encode())
            self._ser.flush()
            reply = self._ser.readline().decode(errors="replace").strip()
        if not reply:
            raise GripperError(
                f"No reply from Arduino (timeout {self._timeout:.0f}s) "
                f"after: {command!r}\n"
                "Check: sketch uploaded? baud=115200? USB cable?"
            )
        if reply.startswith("ERR"):
            raise GripperError(f"Arduino error: {reply}")
        return reply

    def read_raw(self, timeout: float = 3.0) -> str:
        if not self.is_connected:
            raise GripperError("Gripper not connected")
        deadline = time.monotonic() + timeout
        buf = b""
        old_to = self._ser.timeout
        self._ser.timeout = 0.1
        try:
            while time.monotonic() < deadline:
                chunk = self._ser.read(256)
                if chunk:
                    buf += chunk
        finally:
            self._ser.timeout = old_to
        return buf.decode(errors="replace")

    # ── High-level actions ────────────────────────────────────────────────────

    def open_jaws(self) -> str:
        return self._send("OPEN")

    def close_jaws(self) -> str:
        return self._send("CLOSE")

    def set_angle(self, deg: int) -> str:
        """Move servo to an exact angle (0–180°)."""
        deg = max(0, min(180, int(deg)))
        return self._send(f"ANGLE {deg}")

    def grip(self, width_mm: float) -> str:
        """
        Move to the angle corresponding to a jaw opening of width_mm.

        Linearly interpolates between open_angle (max_jaw_mm) and
        close_angle (0 mm).
        """
        ratio = max(0.0, min(1.0, width_mm / self._max_jaw_mm))
        angle = int(self._open_angle + (self._close_angle - self._open_angle) * (1.0 - ratio))
        return self._send(f"ANGLE {angle}")

    def stop(self) -> str:
        """Detach servo (stops jitter, saves power)."""
        return self._send("STOP")

    def status(self) -> int:
        """Return current servo angle in degrees."""
        reply = self._send("STATUS")
        try:
            return int(reply.split()[-1])
        except (ValueError, IndexError):
            raise GripperError(f"Unexpected STATUS reply: {reply!r}")


# ---------------------------------------------------------------------------
# Probe helper
# ---------------------------------------------------------------------------

def probe_gripper(port: str = GripperController.DEFAULT_PORT) -> bool:
    try:
        with GripperController(port) as g:
            g.status()
        return True
    except Exception:
        return False
