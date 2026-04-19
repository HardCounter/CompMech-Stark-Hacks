"""
Export a Dex-Net 2.0 / GQ-CNN 2.0 model to ONNX.

Why this script exists
----------------------
Berkeley's public GQ-CNN 2.0 weights ship as TensorFlow 1.x checkpoints. This
script converts them into a single ``.onnx`` file that plugs straight into
``gripper_cv.heapgrasp.grasp_learned.LearnedGraspScorer`` via
``--grasp-onnx models/gqcnn_2.0.onnx`` (or ``preset="gqcnn2"``).

Two strategies are tried in order. Both produce an ONNX graph with two inputs
(``image`` 4-D depth patch, ``pose`` 2-D [1, 1] gripper depth in metres) and a
``(1, 2)`` softmax output, matching the published GQ-CNN 2.0 spec.

1.  **tf2onnx path**  (requires ``tensorflow==1.15`` + ``tf2onnx``)
    Downloads the TF1 checkpoint from the autolab/gqcnn model zoo and calls
    ``tf2onnx.convert.from_graph_def``. Fastest when TF1 is available.

2.  **PyTorch fallback**  (requires ``torch``)
    Re-implements the GQ-CNN 2.0 architecture in ~80 lines of PyTorch and
    exports it with ``torch.onnx.export``. If ``--checkpoint`` points at a TF1
    checkpoint we try to port the weights tensor-by-tensor; otherwise we export
    the random-initialised network so the rest of the pipeline has a
    structurally-correct model to wire up (accuracy is secondary for the
    hackathon demo — the plan is to fine-tune on Azure as a follow-up).

Usage
-----
    # Preferred: tf2onnx from Berkeley's zoo (needs tensorflow==1.15)
    python scripts/export_gqcnn_onnx.py \\
        --strategy tf2onnx \\
        --output models/gqcnn_2.0.onnx

    # Fallback: PyTorch re-implementation with random init (no deps beyond torch)
    python scripts/export_gqcnn_onnx.py \\
        --strategy pytorch \\
        --output models/gqcnn_2.0.onnx

    # Try tf2onnx, fall back to PyTorch automatically
    python scripts/export_gqcnn_onnx.py --output models/gqcnn_2.0.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


# Berkeley's public Dex-Net 2.0 GQ-CNN model zoo URL.
# This is the canonical mirror as of the Dex-Net 2.0 paper; verify before use.
DEFAULT_GQCNN_ZOO_URL = (
    "https://berkeley.box.com/shared/static/"
    "ez5flo5bkz6u9rdp1z3vuc7ya5nncdas.gz"
)

# GQ-CNN 2.0 canonical hyper-parameters (matches Dex-Net 2.0 paper § 4.2).
IMAGE_INPUT_NAME = "image"
POSE_INPUT_NAME = "pose"
OUTPUT_NAME = "softmax"
PATCH_SIZE = 32
DEPTH_SPAN_M = 0.1
PATCH_SPAN_M = 0.1

# Stored as ONNX metadata so LearnedGraspScorer can auto-detect the preset
# from a well-formed file (see grasp_learned.LearnedGraspScorer.preset="gqcnn2").
ONNX_METADATA = {
    "producer": "gripper_cv.scripts.export_gqcnn_onnx",
    "preset": "gqcnn2",
    "patch_size": str(PATCH_SIZE),
    "patch_span_m": str(PATCH_SPAN_M),
    "depth_span_m": str(DEPTH_SPAN_M),
    "encoding": "metric",
    "image_input": IMAGE_INPUT_NAME,
    "pose_input": POSE_INPUT_NAME,
}


def _attach_metadata(model_path: Path, layout: str) -> None:
    """Stamp ONNX_METADATA plus a layout hint onto the saved ONNX file."""
    import onnx

    model = onnx.load(str(model_path))
    del model.metadata_props[:]
    meta = dict(ONNX_METADATA)
    meta["image_layout"] = layout  # "NCHW" or "NHWC"
    for k, v in meta.items():
        entry = model.metadata_props.add()
        entry.key = k
        entry.value = v
    onnx.save(model, str(model_path))


# ---------------------------------------------------------------------------
# Strategy 1 — tf2onnx
# ---------------------------------------------------------------------------

def export_via_tf2onnx(
    checkpoint_dir: Path,
    output: Path,
) -> None:
    """
    Convert a TF1 GQ-CNN 2.0 checkpoint to ONNX using ``tf2onnx``.

    The autolab/gqcnn checkpoint exposes the frozen graph under
    ``GQ-Image-Wise/model.ckpt``. We rebuild the network with
    ``tensorflow.compat.v1``, restore the checkpoint, and export.
    """
    try:
        import tensorflow.compat.v1 as tf  # type: ignore
        import tf2onnx  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "tf2onnx path requires `tensorflow==1.15` and `tf2onnx`. "
            "Install with:\n"
            "    pip install tensorflow==1.15 tf2onnx"
        ) from exc

    tf.disable_v2_behavior()

    meta_path = _find_checkpoint_meta(checkpoint_dir)
    print(f"[tf2onnx] restoring graph from {meta_path}")

    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(str(meta_path))
        saver.restore(sess, str(meta_path.with_suffix("")))

        graph = sess.graph
        image_in = _find_tensor(graph, ["input_im_node", "image_pl", "Placeholder"])
        pose_in = _find_tensor(graph, ["input_pose_node", "pose_pl", "Placeholder_1"])
        softmax_out = _find_tensor(graph, ["softmax", "Softmax"])

        model_proto, _ = tf2onnx.convert.from_session(
            sess,
            input_names=[image_in.name, pose_in.name],
            output_names=[softmax_out.name],
            opset=13,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as fh:
        fh.write(model_proto.SerializeToString())
    # Dex-Net keeps NHWC layout. The runtime will auto-detect this.
    _attach_metadata(output, layout="NHWC")
    print(f"[tf2onnx] wrote {output}")


def _find_checkpoint_meta(ckpt_dir: Path) -> Path:
    for pattern in ("**/model.ckpt.meta", "**/*.meta"):
        for p in ckpt_dir.glob(pattern):
            return p
    raise FileNotFoundError(
        f"No TF1 checkpoint (.meta) found under {ckpt_dir}. "
        "Download Berkeley's GQ-CNN 2.0 model zoo and unpack it there."
    )


def _find_tensor(graph, candidates):
    for name in candidates:
        try:
            op = graph.get_operation_by_name(name)
            return op.outputs[0]
        except KeyError:
            continue
    raise KeyError(f"None of {candidates} found in graph.")


# ---------------------------------------------------------------------------
# Strategy 2 — PyTorch re-implementation
# ---------------------------------------------------------------------------

def export_via_pytorch(
    output: Path,
    checkpoint_dir: Optional[Path] = None,
) -> None:
    """
    Export a PyTorch re-implementation of GQ-CNN 2.0 to ONNX.

    Architecture (from Dex-Net 2.0 paper § 4.2, Table 3):

        image stream (1 x 32 x 32):
            conv1_1: 7x7, 64 filters, ReLU
            conv1_2: 5x5, 64 filters, ReLU
            pool1:   2x2 max pool
            conv2_1: 3x3, 64 filters, ReLU
            conv2_2: 3x3, 64 filters, ReLU
            flatten
            fc3:     1024, ReLU

        pose stream (1,):
            pc1:     16, ReLU

        merge:
            fc4:     1024, ReLU (on [fc3; pc1])
            fc5:     2 -> softmax

    If ``checkpoint_dir`` is given we attempt to port TF1 weights by name
    (best-effort). Otherwise we export the random-initialised model — the
    pipeline end-to-end still works; only the ranking quality is random.
    """
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyTorch fallback requires `torch`. Install with:\n"
            "    pip install torch"
        ) from exc

    class GQCNN2(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Image stream. Padding chosen so outputs stay manageable on 32x32.
            self.conv1_1 = nn.Conv2d(1, 64, kernel_size=7, padding=3)
            self.conv1_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.fc3 = nn.Linear(64 * 16 * 16, 1024)

            # Pose stream (gripper depth scalar).
            self.pc1 = nn.Linear(1, 16)

            # Merged head.
            self.fc4 = nn.Linear(1024 + 16, 1024)
            self.fc5 = nn.Linear(1024, 2)

            self.relu = nn.ReLU(inplace=True)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, image: "torch.Tensor", pose: "torch.Tensor") -> "torch.Tensor":
            x = self.relu(self.conv1_1(image))
            x = self.relu(self.conv1_2(x))
            x = self.pool1(x)
            x = self.relu(self.conv2_1(x))
            x = self.relu(self.conv2_2(x))
            x = x.flatten(1)
            x = self.relu(self.fc3(x))

            p = self.relu(self.pc1(pose))

            h = torch.cat([x, p], dim=1)
            h = self.relu(self.fc4(h))
            return self.softmax(self.fc5(h))

    model = GQCNN2()
    model.eval()

    if checkpoint_dir is not None:
        ported = _try_port_tf_weights(model, checkpoint_dir)
        print(f"[pytorch] ported {ported} tensors from TF checkpoint")
    else:
        print("[pytorch] exporting random-init network (run Phase B to train)")

    dummy_image = torch.zeros(1, 1, PATCH_SIZE, PATCH_SIZE, dtype=torch.float32)
    dummy_pose = torch.zeros(1, 1, dtype=torch.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    # torch>=2.6 defaults to the TorchDynamo exporter which requires the extra
    # `onnxscript` package. The legacy tracer path still works without it and
    # produces a cleaner graph for this tiny model, so request it explicitly.
    export_kwargs = dict(
        args=(dummy_image, dummy_pose),
        f=str(output),
        input_names=[IMAGE_INPUT_NAME, POSE_INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={
            IMAGE_INPUT_NAME: {0: "batch"},
            POSE_INPUT_NAME: {0: "batch"},
            OUTPUT_NAME: {0: "batch"},
        },
        opset_version=13,
    )
    try:
        torch.onnx.export(model, dynamo=False, **export_kwargs)  # type: ignore[arg-type]
    except TypeError:
        # Older torch (<2.6) does not accept dynamo=; fall back to the default.
        torch.onnx.export(model, **export_kwargs)  # type: ignore[arg-type]
    _attach_metadata(output, layout="NCHW")
    print(f"[pytorch] wrote {output}")


def _try_port_tf_weights(model, ckpt_dir: Path) -> int:
    """Best-effort TF1 → PyTorch weight port. Returns count of copied tensors."""
    try:
        import tensorflow.compat.v1 as tf  # type: ignore
        import torch  # type: ignore
    except ImportError:
        print("[pytorch] tensorflow unavailable — keeping random init")
        return 0

    tf.disable_v2_behavior()
    reader = tf.train.NewCheckpointReader(str(ckpt_dir / "model.ckpt"))
    shape_map = reader.get_variable_to_shape_map()

    name_map = {
        "conv1_1.weight": ["conv1_1/weights", "conv1_1/W"],
        "conv1_1.bias":   ["conv1_1/biases", "conv1_1/b"],
        "conv1_2.weight": ["conv1_2/weights", "conv1_2/W"],
        "conv1_2.bias":   ["conv1_2/biases", "conv1_2/b"],
        "conv2_1.weight": ["conv2_1/weights", "conv2_1/W"],
        "conv2_1.bias":   ["conv2_1/biases", "conv2_1/b"],
        "conv2_2.weight": ["conv2_2/weights", "conv2_2/W"],
        "conv2_2.bias":   ["conv2_2/biases", "conv2_2/b"],
        "fc3.weight":     ["fc3/weights", "fc3/W"],
        "fc3.bias":       ["fc3/biases", "fc3/b"],
        "pc1.weight":     ["pc1/weights", "pc1/W"],
        "pc1.bias":       ["pc1/biases", "pc1/b"],
        "fc4.weight":     ["fc4/weights", "fc4/W"],
        "fc4.bias":       ["fc4/biases", "fc4/b"],
        "fc5.weight":     ["fc5/weights", "fc5/W"],
        "fc5.bias":       ["fc5/biases", "fc5/b"],
    }

    state = model.state_dict()
    copied = 0
    for torch_name, tf_candidates in name_map.items():
        tf_name = next((n for n in tf_candidates if n in shape_map), None)
        if tf_name is None:
            continue
        arr = reader.get_tensor(tf_name)
        tensor = _tf_to_torch_layout(arr, torch_name)
        try:
            state[torch_name].copy_(torch.from_numpy(tensor))
            copied += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [pytorch]   skip {torch_name}: {exc}")
    model.load_state_dict(state)
    return copied


def _tf_to_torch_layout(arr, torch_name: str):
    """Convert TF's HWIO / IO layout to PyTorch's OIHW / OI layout."""
    import numpy as np  # type: ignore

    if torch_name.endswith(".weight") and arr.ndim == 4:
        # TF conv: (H, W, C_in, C_out) → PyTorch: (C_out, C_in, H, W)
        return np.transpose(arr, (3, 2, 0, 1)).copy()
    if torch_name.endswith(".weight") and arr.ndim == 2:
        # TF FC: (in, out) → PyTorch: (out, in)
        return np.transpose(arr, (1, 0)).copy()
    return arr.copy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=["auto", "tf2onnx", "pytorch"],
        default="auto",
        help="Conversion path. 'auto' tries tf2onnx first, then PyTorch.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Directory containing Berkeley's unpacked GQ-CNN 2.0 checkpoint "
             "(for tf2onnx, or optional weight port in the PyTorch path).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/gqcnn_2.0.onnx"),
        help="Output ONNX path (default: models/gqcnn_2.0.onnx).",
    )
    parser.add_argument(
        "--print-zoo-url",
        action="store_true",
        help="Print the default Dex-Net model zoo URL and exit.",
    )
    args = parser.parse_args()

    if args.print_zoo_url:
        print(DEFAULT_GQCNN_ZOO_URL)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.strategy in {"auto", "tf2onnx"} and args.checkpoint is not None:
        try:
            export_via_tf2onnx(args.checkpoint, args.output)
            print(json.dumps({"status": "ok", "strategy": "tf2onnx",
                              "output": str(args.output)}))
            return 0
        except Exception as exc:  # noqa: BLE001
            if args.strategy == "tf2onnx":
                print(f"[tf2onnx] failed: {exc}", file=sys.stderr)
                return 2
            print(f"[tf2onnx] unavailable ({exc}); falling back to PyTorch",
                  file=sys.stderr)

    if args.strategy in {"auto", "pytorch"}:
        try:
            export_via_pytorch(args.output, checkpoint_dir=args.checkpoint)
            print(json.dumps({"status": "ok", "strategy": "pytorch",
                              "output": str(args.output)}))
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"[pytorch] failed: {exc}", file=sys.stderr)
            return 2

    print("No strategy succeeded.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
