"""Tests for the learned grasp scorer (patch render + fallback)."""
import importlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gripper_cv.heapgrasp.grasp import GraspCandidate, pca_grasp, sample_grasps
from gripper_cv.heapgrasp.grasp_learned import (
    LearnedGraspScorer,
    PatchRender,
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

    def test_return_render_wraps_normalized(self):
        pts = _elongated_cloud()
        g = pca_grasp(pts)
        render = render_grasp_patch(pts, g, return_render=True)
        assert isinstance(render, PatchRender)
        assert render.encoding == "normalized"
        assert render.image.shape == (32, 32)
        assert render.image.min() >= 0.0
        assert render.image.max() <= 1.0


class TestRenderGraspPatchMetric:
    def test_metric_returns_patch_render(self):
        pts = _elongated_cloud()
        g = pca_grasp(pts)
        render = render_grasp_patch(pts, g, encoding="metric")
        assert isinstance(render, PatchRender)
        assert render.encoding == "metric"
        assert render.image.shape == (32, 32)
        assert render.image.dtype == np.float32

    def test_metric_has_finite_depth_and_median_fill(self):
        pts = _elongated_cloud()
        g = pca_grasp(pts)
        render = render_grasp_patch(pts, g, encoding="metric", patch_span_m=0.15)
        # No NaNs / infs; median fill must leave the image entirely finite.
        assert np.isfinite(render.image).all()
        # Metric values are offsets from the grasp mid-plane, so they sit
        # inside the patch depth span on both sides of zero.
        half_span = 0.06 / 2.0  # default depth_span_m / 2
        assert render.image.min() >= -half_span - 1e-6
        assert render.image.max() <= half_span + 1e-6

    def test_metric_empty_cloud_is_zero(self):
        g = GraspCandidate((0, 0, 0), (0, 0, -1), (1, 0, 0), 0.05, 0.0)
        render = render_grasp_patch(np.zeros((0, 3)), g, encoding="metric")
        assert isinstance(render, PatchRender)
        assert np.all(render.image == 0.0)
        assert render.gripper_depth_m == 0.0

    def test_metric_fills_empty_cells_with_median(self):
        # Single tight cluster so most cells are empty and should get the
        # median of the filled ones.
        rng = np.random.default_rng(0)
        pts = rng.normal(scale=0.002, size=(500, 3))
        g = GraspCandidate((0, 0, 0), (0, 0, -1), (1, 0, 0), 0.05, 0.0)
        render = render_grasp_patch(
            pts, g, encoding="metric", patch_span_m=0.1, depth_span_m=0.1,
        )
        uniq = np.unique(render.image)
        # Most cells collapse to the single median value; allow for a handful
        # of actual-data cells in the cluster centre.
        assert uniq.size < render.image.size // 2

    def test_metric_gripper_depth_follows_position(self):
        pts = _elongated_cloud()
        g_high = GraspCandidate((0, 0, 0.2), (0, 0, -1), (1, 0, 0), 0.05, 0.0)
        g_low = GraspCandidate((0, 0, 0.0), (0, 0, -1), (1, 0, 0), 0.05, 0.0)
        r_high = render_grasp_patch(pts, g_high, encoding="metric")
        r_low = render_grasp_patch(pts, g_low, encoding="metric")
        # Higher grasp → closer to the nominal camera → smaller depth scalar.
        assert r_high.gripper_depth_m < r_low.gripper_depth_m


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


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None,
    reason="onnxruntime not installed",
)
class TestDualInputOnnxBackend:
    """Covers Dex-Net 2.0 GQ-CNN wiring (image + pose) via onnx.helper stubs."""

    def _build_gqcnn_stub(
        self,
        path: Path,
        *,
        layout: str = "NCHW",
        with_metadata: bool = True,
    ) -> None:
        """Write a tiny two-input ONNX whose output is a softmax(2) over
        (sum(image), pose)."""
        import onnx
        from onnx import TensorProto, helper

        image_shape = (
            [1, 1, 32, 32] if layout == "NCHW" else [1, 32, 32, 1]
        )
        image = helper.make_tensor_value_info(
            "image", TensorProto.FLOAT, image_shape
        )
        pose = helper.make_tensor_value_info(
            "pose", TensorProto.FLOAT, [1, 1]
        )
        softmax_out = helper.make_tensor_value_info(
            "softmax", TensorProto.FLOAT, [1, 2]
        )

        # logits = concat([sum(image), pose*0])  → softmax → (1, 2)
        reduce_node = helper.make_node(
            "ReduceSum", ["image"], ["img_sum"], keepdims=1
        )
        # Flatten image sum to (1, 1) regardless of layout.
        flat = helper.make_node(
            "Flatten", ["img_sum"], ["img_flat"], axis=0
        )
        concat = helper.make_node(
            "Concat", ["img_flat", "pose"], ["logits"], axis=1
        )
        softmax = helper.make_node("Softmax", ["logits"], ["softmax"], axis=1)

        graph = helper.make_graph(
            [reduce_node, flat, concat, softmax],
            "gqcnn_stub",
            [image, pose],
            [softmax_out],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)]
        )
        if with_metadata:
            for k, v in {
                "preset": "gqcnn2",
                "encoding": "metric",
                "patch_size": "32",
                "patch_span_m": "0.1",
                "depth_span_m": "0.1",
                "image_input": "image",
                "pose_input": "pose",
                "image_layout": layout,
            }.items():
                entry = model.metadata_props.add()
                entry.key = k
                entry.value = v
        onnx.save(model, str(path))

    def test_dual_input_end_to_end_nchw(self):
        pytest.importorskip("onnx")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gqcnn_stub.onnx"
            self._build_gqcnn_stub(path, layout="NCHW")
            scorer = LearnedGraspScorer(onnx_path=path, preset="gqcnn2")
            try:
                assert scorer.backend == "onnx"
                assert scorer.encoding == "metric"
                assert scorer.patch_size == 32
                pts = _elongated_cloud()
                g = pca_grasp(pts)
                s = scorer.predict(pts, g)
                assert 0.0 <= s <= 1.0
            finally:
                scorer.close()

    def test_dual_input_auto_preset_from_metadata(self):
        pytest.importorskip("onnx")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gqcnn_stub.onnx"
            self._build_gqcnn_stub(path, layout="NCHW", with_metadata=True)
            scorer = LearnedGraspScorer(onnx_path=path, preset="auto")
            try:
                assert scorer.preset == "gqcnn2"
                assert scorer.encoding == "metric"
            finally:
                scorer.close()

    def test_dual_input_nhwc_layout_transpose(self):
        pytest.importorskip("onnx")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gqcnn_stub_nhwc.onnx"
            self._build_gqcnn_stub(path, layout="NHWC", with_metadata=True)
            scorer = LearnedGraspScorer(onnx_path=path, preset="gqcnn2")
            try:
                pts = _elongated_cloud()
                g = pca_grasp(pts)
                # Should not raise: backend transposes NCHW → NHWC internally.
                s = scorer.predict(pts, g)
                assert 0.0 <= s <= 1.0
            finally:
                scorer.close()

    def test_score_fn_runs_over_multiple_candidates(self):
        pytest.importorskip("onnx")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gqcnn_stub.onnx"
            self._build_gqcnn_stub(path, layout="NCHW")
            scorer = LearnedGraspScorer(onnx_path=path, preset="gqcnn2")
            try:
                pts = _elongated_cloud()
                cands = sample_grasps(pts, n_candidates=8,
                                      score_fn=scorer.score_fn)
                assert len(cands) > 0
                assert all(0.0 <= c.score <= 1.0 for c in cands)
            finally:
                scorer.close()
