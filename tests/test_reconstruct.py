"""Tests for the Shape-from-Silhouette reconstruction core."""
import numpy as np
import pytest
from gripper_cv.heapgrasp.reconstruct import (
    default_camera_matrix,
    shape_from_silhouette,
    voxels_to_pointcloud,
)


class TestCameraMatrix:
    def test_shape(self):
        K = default_camera_matrix()
        assert K.shape == (3, 3)

    def test_bottom_row(self):
        K = default_camera_matrix()
        np.testing.assert_array_equal(K[2], [0.0, 0.0, 1.0])

    def test_principal_point(self):
        K = default_camera_matrix(width=640, height=480)
        assert K[0, 2] == 320.0
        assert K[1, 2] == 240.0

    def test_focal_positive(self):
        K = default_camera_matrix()
        assert K[0, 0] > 0
        assert K[1, 1] > 0

    def test_square_pixels(self):
        K = default_camera_matrix()
        assert K[0, 0] == K[1, 1]

    def test_wider_fov_gives_shorter_focal_length(self):
        K_narrow = default_camera_matrix(fov_deg=40.0)
        K_wide = default_camera_matrix(fov_deg=80.0)
        assert K_narrow[0, 0] > K_wide[0, 0]


class TestShapeFromSilhouette:
    V = 16
    H, W = 240, 320

    @pytest.fixture
    def K(self):
        return default_camera_matrix(self.W, self.H)

    def test_all_white_mask_keeps_all_voxels(self, K):
        mask = np.ones((self.H, self.W), dtype=bool)
        voxels = shape_from_silhouette([mask], [0.0], K, volume_size=self.V)
        assert voxels.all()

    def test_all_black_mask_carves_most_voxels(self, K):
        mask = np.zeros((self.H, self.W), dtype=bool)
        voxels = shape_from_silhouette([mask], [0.0], K, volume_size=self.V)
        # Out-of-frustum voxels survive, but the majority should be carved
        assert voxels.sum() < self.V ** 3 * 0.1

    def test_output_shape(self, K):
        mask = np.ones((self.H, self.W), dtype=bool)
        voxels = shape_from_silhouette([mask], [0.0], K, volume_size=self.V)
        assert voxels.shape == (self.V, self.V, self.V)
        assert voxels.dtype == bool

    def test_output_bool_dtype(self, K):
        mask = np.ones((self.H, self.W), dtype=bool)
        voxels = shape_from_silhouette([mask], [0.0], K, volume_size=self.V)
        assert voxels.dtype == bool

    def test_multiple_views_reduce_occupancy(self, K):
        # Two views from opposite sides should carve more than one view
        mask_partial = np.zeros((self.H, self.W), dtype=bool)
        mask_partial[80:160, 100:220] = True  # central rectangle

        voxels_one = shape_from_silhouette([mask_partial], [0.0], K, volume_size=self.V)
        voxels_two = shape_from_silhouette(
            [mask_partial, mask_partial], [0.0, 90.0], K, volume_size=self.V
        )
        assert voxels_two.sum() <= voxels_one.sum()

    def test_circular_masks_carve_corners(self, K):
        """Circular silhouettes should remove grid corners, preserve center."""
        V = 24
        angles = [0.0, 90.0, 180.0, 270.0]
        masks = []
        for _ in angles:
            mask = np.zeros((self.H, self.W), dtype=bool)
            cy, cx = self.H // 2, self.W // 2
            r = min(self.H, self.W) // 3
            y, x = np.ogrid[:self.H, :self.W]
            mask[(y - cy) ** 2 + (x - cx) ** 2 <= r ** 2] = True
            masks.append(mask)

        voxels = shape_from_silhouette(
            masks, angles, K, volume_size=V, object_diameter=0.15, camera_distance=0.4
        )
        assert not voxels[0, 0, 0], "Grid corner should be carved by circular silhouettes"
        c = V // 2
        assert voxels[c, c, c], "Grid center should survive"


class TestVoxelsToPointcloud:
    def test_full_grid(self):
        V = 8
        voxels = np.ones((V, V, V), dtype=bool)
        pts = voxels_to_pointcloud(voxels, object_diameter=0.16)
        assert pts.shape == (V ** 3, 3)

    def test_empty_grid(self):
        V = 8
        voxels = np.zeros((V, V, V), dtype=bool)
        pts = voxels_to_pointcloud(voxels)
        assert pts.shape == (0, 3)

    def test_single_voxel(self):
        V = 8
        voxels = np.zeros((V, V, V), dtype=bool)
        voxels[0, 0, 0] = True
        pts = voxels_to_pointcloud(voxels, object_diameter=0.16)
        assert pts.shape == (1, 3)

    def test_metric_range(self):
        V = 8
        voxels = np.ones((V, V, V), dtype=bool)
        diameter = 0.2
        pts = voxels_to_pointcloud(voxels, object_diameter=diameter)
        assert pts[:, 0].min() >= -diameter / 2 - 1e-9
        assert pts[:, 0].max() <= diameter / 2 + 1e-9
