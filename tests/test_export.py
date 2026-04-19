"""Tests for PLY, numpy, and grasps.json export."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gripper_cv.heapgrasp.export import (
    save_grasps_json,
    save_masks,
    save_ply,
    save_voxels_npy,
)
from gripper_cv.heapgrasp.grasp import GraspCandidate, grasps_to_json


class TestSavePly:
    def test_creates_file(self):
        pts = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.ply"
            save_ply(pts, path)
            assert path.exists()

    def test_header_vertex_count(self):
        pts = np.random.rand(7, 3).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pts.ply"
            save_ply(pts, path)
            content = path.read_text()
        assert "element vertex 7" in content
        assert "end_header" in content

    def test_with_colors(self):
        pts = np.array([[0.0, 0.0, 0.0]])
        colors = np.array([[255, 128, 0]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "colored.ply"
            save_ply(pts, path, colors=colors)
            content = path.read_text()
        assert "property uchar red" in content
        assert "property uchar green" in content
        assert "property uchar blue" in content

    def test_creates_parent_dir(self):
        pts = np.zeros((2, 3))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deep" / "nested" / "out.ply"
            save_ply(pts, path)
            assert path.exists()

    def test_point_values_in_file(self):
        pts = np.array([[1.5, 2.5, 3.5]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vals.ply"
            save_ply(pts, path)
            content = path.read_text()
        assert "1.500000" in content
        assert "2.500000" in content
        assert "3.500000" in content


class TestSaveVoxelsNpy:
    def test_roundtrip(self):
        voxels = np.random.rand(16, 16, 16) > 0.5
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voxels.npy"
            save_voxels_npy(voxels, path)
            loaded = np.load(path)
        assert np.array_equal(voxels, loaded)

    def test_creates_parent_dir(self):
        voxels = np.zeros((4, 4, 4), dtype=bool)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "voxels.npy"
            save_voxels_npy(voxels, path)
            assert path.exists()


class TestSaveMasks:
    def test_saves_correct_count(self):
        masks = [np.zeros((64, 64), dtype=bool) for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "masks"
            save_masks(masks, out)
            pngs = list(out.glob("*.png"))
        assert len(pngs) == 3

    def test_mask_filenames(self):
        masks = [np.zeros((32, 32), dtype=bool) for _ in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "masks"
            save_masks(masks, out)
            names = {p.name for p in out.glob("*.png")}
        assert "mask_000.png" in names
        assert "mask_001.png" in names


class TestSaveGraspsJson:
    def _sample_payload(self) -> dict:
        g = GraspCandidate(
            position=(0.01, 0.02, 0.03),
            approach=(0.0, 0.0, -1.0),
            jaw_axis=(1.0, 0.0, 0.0),
            width=0.05,
            score=0.75,
        )
        return grasps_to_json([g], extra={"views": 8})

    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grasps.json"
            save_grasps_json(self._sample_payload(), path)
            assert path.exists()

    def test_round_trip(self):
        payload = self._sample_payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grasps.json"
            save_grasps_json(payload, path)
            decoded = json.loads(path.read_text())
        assert decoded["frame"] == "turntable_world"
        assert decoded["grasps"][0]["score"] == pytest.approx(0.75)
        assert decoded["meta"]["views"] == 8

    def test_creates_parent_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "deep" / "grasps.json"
            save_grasps_json(self._sample_payload(), path)
            assert path.exists()
