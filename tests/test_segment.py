"""Tests for silhouette extraction (background subtraction path)."""
import numpy as np
import pytest
from gripper_cv.heapgrasp.capture import CaptureSession
from gripper_cv.heapgrasp.segment import extract_silhouettes


def _solid_rgb(h, w, color=(128, 128, 128)):
    return np.full((h, w, 3), color, dtype=np.uint8)


def _frame_with_circle(h, w, bg_color=(50, 50, 50), obj_color=(200, 200, 200)):
    frame = _solid_rgb(h, w, bg_color)
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    frame[(y - cy) ** 2 + (x - cx) ** 2 <= r ** 2] = obj_color
    return frame


class TestBackgroundSubtraction:
    H, W = 120, 160

    def test_identical_frames_give_empty_mask(self):
        bg = _solid_rgb(self.H, self.W, (100, 100, 100))
        session = CaptureSession(frames=[bg.copy()], angles_deg=[0.0], background=bg.copy())
        masks = extract_silhouettes(session, method="background", bg_threshold=10)
        assert len(masks) == 1
        assert not masks[0].any(), "Identical frames should produce empty mask"

    def test_bright_circle_detected(self):
        bg = _solid_rgb(self.H, self.W, (30, 30, 30))
        frame = _frame_with_circle(self.H, self.W, bg_color=(30, 30, 30), obj_color=(200, 200, 200))
        session = CaptureSession(frames=[frame], angles_deg=[0.0], background=bg.copy())
        masks = extract_silhouettes(session, method="background", bg_threshold=10)
        mask = masks[0]
        # Center pixel should be foreground
        cy, cx = self.H // 2, self.W // 2
        assert mask[cy, cx], "Circle center should be detected as foreground"

    def test_returns_bool_masks(self):
        bg = _solid_rgb(self.H, self.W)
        frame = _frame_with_circle(self.H, self.W)
        session = CaptureSession(frames=[frame], angles_deg=[0.0], background=bg)
        masks = extract_silhouettes(session, method="background")
        assert masks[0].dtype == bool

    def test_mask_count_matches_frames(self):
        bg = _solid_rgb(self.H, self.W, (30, 30, 30))
        frames = [_frame_with_circle(self.H, self.W) for _ in range(4)]
        angles = [0.0, 90.0, 180.0, 270.0]
        session = CaptureSession(frames=frames, angles_deg=angles, background=bg)
        masks = extract_silhouettes(session, method="background")
        assert len(masks) == 4

    def test_missing_background_raises(self):
        frame = _solid_rgb(self.H, self.W)
        session = CaptureSession(frames=[frame], angles_deg=[0.0], background=None)
        with pytest.raises(ValueError, match="Background frame missing"):
            extract_silhouettes(session, method="background")

    def test_unknown_method_raises(self):
        bg = _solid_rgb(self.H, self.W)
        session = CaptureSession(frames=[bg.copy()], angles_deg=[0.0], background=bg)
        with pytest.raises(ValueError, match="Unknown segmentation method"):
            extract_silhouettes(session, method="notamethod")

    def test_higher_threshold_less_sensitive(self):
        bg = _solid_rgb(self.H, self.W, (100, 100, 100))
        # Slight change: only +20 per channel
        frame = _solid_rgb(self.H, self.W, (120, 120, 120))
        session = CaptureSession(frames=[frame], angles_deg=[0.0], background=bg)

        masks_sensitive = extract_silhouettes(session, method="background", bg_threshold=10)
        masks_strict = extract_silhouettes(session, method="background", bg_threshold=30)
        # Stricter threshold should detect less (or equal) foreground
        assert masks_strict[0].sum() <= masks_sensitive[0].sum()
