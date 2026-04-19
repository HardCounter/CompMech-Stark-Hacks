"""Tests for the shared grasp module."""
import json

import numpy as np
import pytest

from gripper_cv.heapgrasp.grasp import (
    GraspCandidate,
    ObjectShape,
    format_grasp_plan,
    grasp_from_voxels,
    grasps_to_json,
    pca_grasp,
    sample_grasps,
)


def _elongated_cloud(n: int = 800, seed: int = 0) -> np.ndarray:
    """A thin cylinder along X — long axis should align with X."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.06, 0.06, n)   # 12 cm long
    y = rng.normal(0.0, 0.010, n)     # ~2 cm tall
    z = rng.normal(0.0, 0.005, n)     # ~1 cm deep
    return np.stack([x, y, z], axis=1)


def _sphere_cloud(n: int = 800, radius: float = 0.04, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts * rng.uniform(0.0, radius, size=(n, 1))


class TestObjectShape:
    def test_pca_recovers_dominant_axis(self):
        pts = _elongated_cloud()
        shape = ObjectShape.from_points(pts)
        assert abs(shape.long_axis[0]) > 0.9
        assert shape.extents_m[0] > shape.extents_m[1]
        assert shape.extents_m[1] >= shape.extents_m[2]

    def test_needs_enough_points(self):
        with pytest.raises(ValueError):
            ObjectShape.from_points(np.zeros((2, 3)))


class TestPcaGrasp:
    def test_position_near_centroid(self):
        pts = _elongated_cloud()
        g = pca_grasp(pts)
        assert np.linalg.norm(np.asarray(g.position) - pts.mean(0)) < 1e-6

    def test_jaw_axis_unit(self):
        pts = _elongated_cloud()
        g = pca_grasp(pts)
        assert abs(np.linalg.norm(g.jaw_axis) - 1.0) < 1e-6

    def test_approach_unit(self):
        pts = _elongated_cloud()
        g = pca_grasp(pts)
        assert abs(np.linalg.norm(g.approach) - 1.0) < 1e-6

    def test_width_capped(self):
        pts = _elongated_cloud()
        g = pca_grasp(pts, max_width_m=0.08)
        assert g.width <= 0.08

    def test_score_in_unit_interval(self):
        pts = _elongated_cloud()
        g = pca_grasp(pts)
        assert 0.0 <= g.score <= 1.0


class TestSampleGrasps:
    def test_returns_candidates_for_elongated_shape(self):
        pts = _elongated_cloud()
        cands = sample_grasps(pts, n_candidates=32)
        assert len(cands) > 0

    def test_sorted_by_score(self):
        pts = _elongated_cloud()
        cands = sample_grasps(pts, n_candidates=64)
        scores = [c.score for c in cands]
        assert scores == sorted(scores, reverse=True)

    def test_widths_bounded(self):
        pts = _elongated_cloud()
        cands = sample_grasps(pts, n_candidates=32, max_width_m=0.09)
        assert all(c.width <= 0.09 for c in cands)

    def test_rng_reproducible(self):
        pts = _elongated_cloud()
        a = sample_grasps(pts, n_candidates=32, rng=np.random.default_rng(7))
        b = sample_grasps(pts, n_candidates=32, rng=np.random.default_rng(7))
        assert [c.position for c in a] == [c.position for c in b]

    def test_rejects_tiny_cloud(self):
        with pytest.raises(ValueError):
            sample_grasps(np.zeros((3, 3)))

    def test_custom_score_function_is_used(self):
        pts = _elongated_cloud()
        calls = {"n": 0}

        def constant(_: GraspCandidate, __: np.ndarray) -> float:
            calls["n"] += 1
            return 0.5

        cands = sample_grasps(pts, n_candidates=16, score_fn=constant)
        assert calls["n"] == len(cands)
        assert all(c.score == 0.5 for c in cands)


class TestGraspFromVoxels:
    def test_empty_voxels(self):
        voxels = np.zeros((8, 8, 8), dtype=bool)
        assert grasp_from_voxels(voxels, object_diameter=0.15) == []

    def test_top_k_limit(self):
        V = 16
        voxels = np.zeros((V, V, V), dtype=bool)
        # Elongated block along X
        voxels[2:14, 6:10, 7:9] = True
        out = grasp_from_voxels(voxels, object_diameter=0.15, top_k=3)
        assert 1 <= len(out) <= 3

    def test_first_candidate_is_deterministic_pca(self):
        V = 16
        voxels = np.zeros((V, V, V), dtype=bool)
        voxels[2:14, 6:10, 7:9] = True
        a = grasp_from_voxels(voxels, object_diameter=0.15, top_k=1)[0]
        b = grasp_from_voxels(voxels, object_diameter=0.15, top_k=1)[0]
        assert a == b


class TestGraspsToJson:
    def test_round_trip_json(self):
        g = GraspCandidate(
            position=(0.0, 0.01, 0.02),
            approach=(0.0, 0.0, -1.0),
            jaw_axis=(1.0, 0.0, 0.0),
            width=0.05,
            score=0.8,
        )
        payload = grasps_to_json([g], extra={"views": 8})
        encoded = json.dumps(payload)
        decoded = json.loads(encoded)
        assert decoded["frame"] == "turntable_world"
        assert decoded["meta"]["views"] == 8
        assert decoded["grasps"][0]["width_m"] == pytest.approx(0.05)
        assert decoded["grasps"][0]["position_m"] == [0.0, 0.01, 0.02]

    def test_contains_convention_block(self):
        doc = grasps_to_json([])
        assert doc["convention"]["x"] == "right"
        assert doc["convention"]["y"] == "up"


class TestFormatGraspPlan:
    def test_message_for_empty_voxels(self):
        voxels = np.zeros((8, 8, 8), dtype=bool)
        text = format_grasp_plan(voxels, object_diameter=0.15,
                                 camera_distance=0.4, n_views=4)
        assert "Insufficient point cloud data" in text

    def test_plan_for_filled_voxels(self):
        V = 16
        voxels = np.zeros((V, V, V), dtype=bool)
        voxels[3:13, 6:10, 7:9] = True
        text = format_grasp_plan(voxels, object_diameter=0.15,
                                 camera_distance=0.4, n_views=8)
        assert "THEORETICAL GRASP PLAN" in text
        assert "OBJECT SUMMARY" in text
        assert "SAFETY NOTES" in text
