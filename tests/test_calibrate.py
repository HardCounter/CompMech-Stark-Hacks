"""Tests for auto-calibration utilities."""
import warnings
import numpy as np
import pytest
from gripper_cv.heapgrasp.calibrate import estimate_object_diameter
from gripper_cv.heapgrasp.reconstruct import default_camera_matrix


class TestEstimateObjectDiameter:
    @pytest.fixture
    def K(self):
        return default_camera_matrix(640, 480)  # fx ≈ 554

    def test_known_size_approximately_correct(self, K):
        fx = K[0, 0]
        # A 100px-wide object at 0.5 m should back-project to ~0.09 m
        # With 5% padding: ~0.095 m
        mask = np.zeros((480, 640), dtype=bool)
        mask[200:300, 270:370] = True  # 100 × 100 px square
        diameter = estimate_object_diameter([mask], K, camera_distance=0.5)
        expected = 100 * 0.5 / fx * 1.05
        assert abs(diameter - expected) < 0.005

    def test_empty_mask_returns_fallback(self, K):
        mask = np.zeros((480, 640), dtype=bool)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diameter = estimate_object_diameter([mask], K, 0.4, fallback_diameter=0.15)
        assert diameter == 0.15
        assert any("empty" in str(warning.message).lower() for warning in w)

    def test_larger_mask_gives_larger_diameter(self, K):
        mask_small = np.zeros((480, 640), dtype=bool)
        mask_small[220:260, 300:340] = True  # 40 px

        mask_large = np.zeros((480, 640), dtype=bool)
        mask_large[180:300, 260:380] = True  # 120 px

        d_small = estimate_object_diameter([mask_small], K, 0.4)
        d_large = estimate_object_diameter([mask_large], K, 0.4)
        assert d_large > d_small

    def test_max_across_views(self, K):
        """Diameter should be the max across all provided views."""
        mask_small = np.zeros((480, 640), dtype=bool)
        mask_small[230:250, 310:330] = True  # 20 px

        mask_big = np.zeros((480, 640), dtype=bool)
        mask_big[100:380, 100:540] = True  # 440 px wide

        d_multi = estimate_object_diameter([mask_small, mask_big], K, 0.4)
        d_big_only = estimate_object_diameter([mask_big], K, 0.4)
        assert abs(d_multi - d_big_only) < 1e-9

    def test_padding_applied(self, K):
        mask = np.zeros((480, 640), dtype=bool)
        mask[200:300, 270:370] = True
        fx = K[0, 0]
        raw = 100 * 0.5 / fx
        d = estimate_object_diameter([mask], K, 0.5, padding=1.10)
        assert abs(d - raw * 1.10) < 0.001
