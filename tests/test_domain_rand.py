"""Tests for domain randomization utilities (no torch required)."""
import numpy as np
import pytest
from gripper_cv.sim2real.domain_rand import (
    BackgroundRandomizer,
    DomainRandomTransform,
    random_background,
)


def _solid_image(H, W, color=(100, 100, 100)):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = color
    return img


class TestRandomBackground:
    def test_shape(self):
        bg = random_background((120, 160))
        assert bg.shape == (120, 160, 3)

    def test_dtype(self):
        bg = random_background((64, 64))
        assert bg.dtype == np.uint8

    def test_value_range(self):
        bg = random_background((64, 64))
        assert bg.min() >= 0
        assert bg.max() <= 255

    def test_reproducible_with_same_rng(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        bg1 = random_background((64, 64), rng1)
        bg2 = random_background((64, 64), rng2)
        np.testing.assert_array_equal(bg1, bg2)


class TestBackgroundRandomizer:
    H, W = 80, 100

    def test_output_shape_preserved(self):
        br = BackgroundRandomizer()
        img = _solid_image(self.H, self.W, (200, 200, 200))
        mask = np.zeros((self.H, self.W), dtype=bool)
        out = br(img, mask)
        assert out.shape == (self.H, self.W, 3)

    def test_foreground_pixels_unchanged(self):
        br = BackgroundRandomizer()
        img = _solid_image(self.H, self.W, (200, 200, 200))
        # Mark top-left quarter as foreground
        mask = np.zeros((self.H, self.W), dtype=bool)
        mask[: self.H // 2, : self.W // 2] = True
        out = br(img, mask)
        np.testing.assert_array_equal(out[mask], img[mask])

    def test_background_pixels_changed(self):
        br = BackgroundRandomizer()
        # Solid very specific colour unlikely to appear by chance
        img = _solid_image(self.H, self.W, (7, 7, 7))
        mask = np.zeros((self.H, self.W), dtype=bool)  # all background
        out = br(img, mask)
        # With all-background mask, output should differ from input
        assert not np.array_equal(out, img)

    def test_all_foreground_mask_leaves_image_unchanged(self):
        br = BackgroundRandomizer()
        img = _solid_image(self.H, self.W, (150, 100, 50))
        mask = np.ones((self.H, self.W), dtype=bool)  # all foreground
        out = br(img, mask)
        np.testing.assert_array_equal(out, img)

    def test_int64_mask_treated_same_as_bool(self):
        br = BackgroundRandomizer()
        img = _solid_image(self.H, self.W, (200, 200, 200))
        bool_mask = np.ones((self.H, self.W), dtype=bool)
        int_mask = np.ones((self.H, self.W), dtype=np.int64)
        out_bool = br(img, bool_mask)
        out_int = br(img, int_mask)
        np.testing.assert_array_equal(out_bool, out_int)


class TestDomainRandomTransform:
    H, W = 64, 80

    def test_output_shapes_preserved(self):
        tr = DomainRandomTransform()
        img = _solid_image(self.H, self.W)
        mask = np.zeros((self.H, self.W), dtype=np.int64)
        out_img, out_mask = tr(img, mask)
        assert out_img.shape == (self.H, self.W, 3)
        assert out_mask.shape == (self.H, self.W)

    def test_output_image_dtype(self):
        tr = DomainRandomTransform()
        img = _solid_image(self.H, self.W)
        mask = np.zeros((self.H, self.W), dtype=np.int64)
        out_img, _ = tr(img, mask)
        assert out_img.dtype == np.uint8

    def test_output_pixel_range(self):
        tr = DomainRandomTransform()
        img = _solid_image(self.H, self.W)
        mask = np.zeros((self.H, self.W), dtype=np.int64)
        out_img, _ = tr(img, mask)
        assert out_img.min() >= 0
        assert out_img.max() <= 255

    def test_mask_values_preserved(self):
        """Class index values in the mask should never change."""
        tr = DomainRandomTransform(p_bg_replace=0.0, p_color_jitter=0.0,
                                   p_noise=0.0, p_flip=0.0)
        img = _solid_image(self.H, self.W)
        mask = np.zeros((self.H, self.W), dtype=np.int64)
        mask[10:30, 10:40] = 1
        mask[40:60, 40:70] = 2
        _, out_mask = tr(img, mask)
        np.testing.assert_array_equal(out_mask, mask)

    def test_flip_with_prob_one_mirrors_image(self):
        tr = DomainRandomTransform(
            p_bg_replace=0.0, p_color_jitter=0.0, p_noise=0.0, p_flip=1.0
        )
        img = np.arange(self.H * self.W * 3, dtype=np.uint8).reshape(self.H, self.W, 3)
        mask = np.zeros((self.H, self.W), dtype=np.int64)
        out_img, out_mask = tr(img, mask)
        np.testing.assert_array_equal(out_img, img[:, ::-1])
        np.testing.assert_array_equal(out_mask, mask[:, ::-1])

    def test_flip_with_prob_zero_leaves_image_unchanged(self):
        tr = DomainRandomTransform(
            p_bg_replace=0.0, p_color_jitter=0.0, p_noise=0.0, p_flip=0.0
        )
        img = _solid_image(self.H, self.W, (77, 88, 99))
        mask = np.zeros((self.H, self.W), dtype=np.int64)
        out_img, _ = tr(img, mask)
        np.testing.assert_array_equal(out_img, img)

    def test_no_ops_leave_mask_intact(self):
        tr = DomainRandomTransform(
            p_bg_replace=0.0, p_color_jitter=0.0, p_noise=0.0, p_flip=0.0
        )
        mask = np.random.default_rng(0).integers(0, 4, size=(self.H, self.W)).astype(np.int64)
        img = _solid_image(self.H, self.W)
        _, out_mask = tr(img, mask)
        np.testing.assert_array_equal(out_mask, mask)
