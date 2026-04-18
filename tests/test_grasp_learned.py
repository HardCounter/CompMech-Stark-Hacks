"""Tests for the learned grasp scorer (patch render + fallback)."""
import importlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gripper_cv.heapgrasp.grasp import GraspCandidate, pca_grasp, sample_grasps
from gripper_cv.heapgrasp.grasp_learned import (
    LearnedGraspScorer,
    render_grasp_patch,
)


def _elongated_cloud(n: int = 800, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.06, 0.06, n)
    y = rng.normal(0.0, 0.010, n)
    z = rng.normal(0.0, 0.005, n)
    return np.stack([x, y, z], axis=1)


class TestRenderGraspPatch:
    def test_zero_patch_for_empty_cloud(self):
        g = GraspCandidate((0, 0, 0), (0, 0, -1), (1, 0, 0), 0.05, 0.0)
        patch = render_grasp_patch(np.zeros((0, 3)), g)
        assert patch.shape == (32, 32)
        assert patch.sum() == 0

    def test_patch_values_in_unit_interval(self):
        pts = _elongated_cloud()
        g = pca_grasp(pts)
        patch = render_grasp_patch(pts, g)
        assert patch.shape == (32, 32)
        assert patch.min() >= 0.0
        assert patch.max() <= 1.0

    def test_patch_has_signal_near_elongated_object(self):
        pts = _elongated_cloud()
        g = pca_grasp(pts)
        patch = render_grasp_patch(pts, g, patch_span_m=0.15)
        assert patch.sum() > 0

    def test_custom_patch_size(self):
        pts = _elongated_cloud()
        g = pca_grasp(pts)
        patch = render_grasp_patch(pts, g, patch_size=16)
        assert patch.shape == (16, 16)


class TestLearnedGraspScorerFallback:
    def test_fallback_backend_label(self):
        scorer = LearnedGraspScorer()
        assert scorer.backend == "fallback"

    def test_fallback_predicts_default_score(self):
        pts = _elongated_cloud()
        g = pca_grasp(pts)
        scorer = LearnedGraspScorer()
        s = scorer.predict(pts, g)
        assert 0.0 <= s <= 1.0

    def test_score_fn_integrates_with_sample_grasps(self):
        pts = _elongated_cloud()
        scorer = LearnedGraspScorer()
        cands = sample_grasps(pts, n_candidates=16, score_fn=scorer.score_fn)
        assert len(cands) > 0
        assert all(0.0 <= c.score <= 1.0 for c in cands)

    def test_context_manager_closes(self):
        with LearnedGraspScorer() as scorer:
            assert scorer.backend == "fallback"


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None,
    reason="onnxruntime not installed",
)
class TestOnnxBackend:
    def _build_identity_onnx(self, path: Path) -> None:
        """Write a tiny ONNX model whose output is sigmoid(sum(input))."""
        import onnx
        from onnx import TensorProto, helper

        x = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, 1, 32, 32]
        )
        y = helper.make_tensor_value_info("score", TensorProto.FLOAT, [1])

        reduce_node = helper.make_node(
            "ReduceSum", ["input"], ["sum"], keepdims=0
        )
        sigmoid_node = helper.make_node("Sigmoid", ["sum"], ["score"])

        graph = helper.make_graph(
            [reduce_node, sigmoid_node],
            "grasp_stub",
            [x],
            [y],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        onnx.save(model, str(path))

    def test_end_to_end(self):
        onnx = pytest.importorskip("onnx")  # noqa: F841
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stub.onnx"
            self._build_identity_onnx(path)
            scorer = LearnedGraspScorer(onnx_path=path)
            try:
                assert scorer.backend == "onnx"
                pts = _elongated_cloud()
                g = pca_grasp(pts)
                s = scorer.predict(pts, g)
                assert 0.0 <= s <= 1.0
            finally:
                scorer.close()
