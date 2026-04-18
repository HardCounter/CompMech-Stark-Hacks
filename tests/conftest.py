"""
Stub out picamera2 so heapgrasp modules can be imported on non-Pi machines.
Must run before any gripper_cv imports.
"""
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("picamera2", MagicMock())
