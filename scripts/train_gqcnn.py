"""
Phase B: train / fine-tune a GQ-CNN 2.0-style network on Dex-Net 2.0 data.

This script is a **skeleton** — it is not meant to be run as-is during the
hackathon demo. It is checked in so the Azure training pipeline is already
scaffolded when GPU credits come online. Fill in the ``TODO`` markers when
you connect the Azure VM.

Why this is deferred
--------------------
Phase A already gives us a working ML grasp-scoring pipeline with a random-
init network. Accuracy is secondary for the hackathon demo. Once an Azure
GPU VM is online we can close the domain gap here.

Dataset
-------
Dex-Net 2.0 publishes its training tensors at:

    https://berkeley.app.box.com/v/dex-net-2-tensor-dataset

The full set is ~230 GB (raw depth images + grasps + quality labels).
The Berkeley autolab/gqcnn repo ships a helper to download it. On Azure:

    az login
    az storage blob download-batch \\
        --source dexnet2 \\
        --destination /mnt/data/dexnet2 \\
        --account-name <your-account>

Or via the public Box mirror:

    # Requires box-sdk or manual download; see the Dex-Net paper for the URL.
    python scripts/download_dexnet2.py --dest /mnt/data/dexnet2

Training target
---------------
Reproduce the paper's ~85% validation accuracy on a 10% subset for the first
run (fast sanity check). Output: ``models/gqcnn_2.0_finetuned.onnx`` with
the same ONNX metadata (``preset=gqcnn2``) as the export script so
``--grasp-preset auto`` just works.

Architecture
------------
Identical to the one in ``scripts/export_gqcnn_onnx.py`` (PyTorch path). We
re-import it here so a single source of truth drives both export and
training.

Usage (once Azure is online)
----------------------------

    python scripts/train_gqcnn.py \\
        --data-dir /mnt/data/dexnet2 \\
        --output models/gqcnn_2.0_finetuned.onnx \\
        --epochs 5 \\
        --batch-size 128

    # Fine-tune on patches rendered from our own visual hull instead
    python scripts/train_gqcnn.py \\
        --data-dir outputs/heapgrasp_labeled \\
        --source-format visual_hull \\
        --pretrained models/gqcnn_2.0.onnx \\
        --output models/gqcnn_2.0_ours.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_gqcnn2():
    """Lazy import to avoid torch cost when --help is called."""
    import torch.nn as nn  # type: ignore

    class GQCNN2(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1_1 = nn.Conv2d(1, 64, kernel_size=7, padding=3)
            self.conv1_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.fc3 = nn.Linear(64 * 16 * 16, 1024)
            self.pc1 = nn.Linear(1, 16)
            self.fc4 = nn.Linear(1024 + 16, 1024)
            self.fc5 = nn.Linear(1024, 2)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, image, pose):
            import torch  # type: ignore
            x = self.relu(self.conv1_1(image))
            x = self.relu(self.conv1_2(x))
            x = self.pool1(x)
            x = self.relu(self.conv2_1(x))
            x = self.relu(self.conv2_2(x))
            x = x.flatten(1)
            x = self.relu(self.fc3(x))
            p = self.relu(self.pc1(pose))
            h = self.relu(self.fc4(torch.cat([x, p], dim=1)))
            return self.fc5(h)  # raw logits; CE loss does softmax

    return GQCNN2


def build_dataset(data_dir: Path, source_format: str):
    """
    TODO (Phase B): implement dataset loading.

    For ``source_format='dexnet2'``:
        Dex-Net 2.0 tensors live in ``${data_dir}/tensors/{depth_ims, grasps,
        labels}_NNNNN.npz``. Each tensor file holds 1000 samples of:
          - depth_ims: (1000, 32, 32) float32 (metres, centred)
          - hand_poses: (1000, 6)   grasp pose; index 2 is gripper depth
          - robust_ferrari_canny: (1000,) quality in [0, 1]
        Label = 1 if quality > 0.002 (paper's epsilon threshold).

    For ``source_format='visual_hull'``:
        Use ``gripper_cv.heapgrasp.grasp_learned.render_grasp_patch(...,
        encoding='metric')`` on our own PLY outputs, with success labels from
        real gripper trials.

    This function should return a torch Dataset with __getitem__ yielding
    ``(image_1x32x32, pose_1, label_int)``.
    """
    raise NotImplementedError("Phase B: implement once Azure + data are wired.")


def train(args: argparse.Namespace) -> None:
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        from torch.utils.data import DataLoader  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Phase B training requires torch. Install with:\n"
            "    pip install torch torchvision"
        ) from exc

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"[train] device = {device}")

    GQCNN2 = _build_gqcnn2()
    model = GQCNN2().to(device)

    if args.pretrained is not None:
        # TODO (Phase B): load weights from an ONNX or a torch checkpoint.
        print(f"[train] (stub) would warm-start from {args.pretrained}")

    dataset = build_dataset(args.data_dir, args.source_format)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for step, (image, pose, label) in enumerate(loader):
            image = image.to(device)
            pose = pose.to(device)
            label = label.to(device)
            logits = model(image, pose)
            loss = criterion(logits, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % args.log_every == 0:
                print(f"  epoch={epoch} step={step} loss={loss.item():.4f}")

    _export_trained_to_onnx(model, args.output)
    print(f"[train] wrote {args.output}")


def _export_trained_to_onnx(model, output: Path) -> None:
    """Reuse the Phase A exporter so the ONNX metadata stays consistent."""
    import torch  # type: ignore

    from scripts import export_gqcnn_onnx as exporter  # type: ignore

    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_image = torch.zeros(1, 1, 32, 32)
    dummy_pose = torch.zeros(1, 1)
    torch.onnx.export(
        model, (dummy_image, dummy_pose), str(output),
        input_names=[exporter.IMAGE_INPUT_NAME, exporter.POSE_INPUT_NAME],
        output_names=[exporter.OUTPUT_NAME],
        opset_version=13,
        dynamo=False,  # legacy exporter; see comment in export_gqcnn_onnx.py
    )
    exporter._attach_metadata(output, layout="NCHW")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory with Dex-Net 2.0 tensors "
                             "(or visual-hull patches).")
    parser.add_argument("--source-format",
                        choices=["dexnet2", "visual_hull"],
                        default="dexnet2")
    parser.add_argument("--output", type=Path,
                        default=Path("models/gqcnn_2.0_finetuned.onnx"))
    parser.add_argument("--pretrained", type=Path, default=None,
                        help="Optional warm-start checkpoint.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU training (not recommended).")
    args = parser.parse_args()

    try:
        train(args)
    except NotImplementedError as exc:
        print(f"[train] {exc}", file=sys.stderr)
        print(
            "[train] This is the Phase B skeleton — see the module docstring "
            "for the concrete steps to implement once Azure is online.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
