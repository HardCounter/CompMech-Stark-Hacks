"""
Fine-tune DeepLabv3+ (ResNet-50) for 4-class transparent-object segmentation.

Label schema:
  0 = background
  1 = transparent object
  2 = specular (mirror-like) object
  3 = opaque object

The pretrained COCO head (21 classes) is replaced with a 4-class conv.
The ResNet-50 backbone uses a 10× lower learning rate than the head.

Usage
-----
  # Smoke-test with synthetic data (no dataset required):
  gripper-cv-train-seg --synthetic-length 100 --epochs 2 --output-dir /tmp/seg

  # Fine-tune on a Transpose-format dataset:
  gripper-cv-train-seg --data-dir /data/transpose --epochs 20 --output-dir outputs/seg

  # Use finetuned model in HEAPGrasp pipeline:
  gripper-cv-heapgrasp --segment finetuned --seg-checkpoint outputs/seg/best_model.pt

All torch imports are lazy (inside function bodies only).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gripper_cv.training.dataset import NUM_CLASSES


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_seg_model(num_classes: int = NUM_CLASSES, pretrained: bool = True, device: str = "cpu"):
    """
    Build DeepLabv3+ (ResNet-50) with a custom head for num_classes.

    The COCO pretrained backbone is kept; only the classifier head is replaced.
    Lazy: imports torch and torchvision inside.

    Args:
        num_classes: number of output classes (default 4 — HEAPGrasp schema)
        pretrained:  use COCO pretrained backbone weights (recommended)
        device:      torch device string

    Returns:
        model in train mode on the requested device
    """
    import torch
    import torch.nn as nn
    from torchvision.models.segmentation import (
        DeepLabV3_ResNet50_Weights,
        deeplabv3_resnet50,
    )

    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet50(weights=weights)

    # Replace main classifier head
    in_ch = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    # Replace auxiliary classifier head (used during training only)
    if model.aux_classifier is not None:
        in_ch_aux = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(in_ch_aux, num_classes, kernel_size=1)

    return model.to(torch.device(device))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_miou(
    preds: np.ndarray, targets: np.ndarray, num_classes: int = NUM_CLASSES
) -> float:
    """Mean Intersection-over-Union across all classes present in targets."""
    ious = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        tgt_cls = targets == cls
        inter = int((pred_cls & tgt_cls).sum())
        union = int((pred_cls | tgt_cls).sum())
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_seg(
    train_dataset,
    val_dataset,
    output_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cpu",
    num_classes: int = NUM_CLASSES,
    num_workers: int = 0,
) -> None:
    """
    Fine-tune DeepLabv3+ and save the best checkpoint by validation mIoU.

    Args:
        train_dataset, val_dataset: torch Dataset objects
        output_dir:    directory for checkpoints and training history
        epochs:        number of training epochs
        batch_size:    samples per batch
        lr:            base learning rate (backbone uses lr * 0.1)
        device:        "cpu" or "cuda"
        num_classes:   number of segmentation classes
        num_workers:   DataLoader worker processes (0 = main process)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device)
    use_amp = dev.type == "cuda"
    model = build_seg_model(num_classes=num_classes, pretrained=True, device=device)

    # Differential learning rates: backbone at lr * 0.1, head at lr
    backbone_params = [
        p for n, p in model.named_parameters()
        if "backbone" in n and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if "backbone" not in n and p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": lr * 0.1},
         {"params": head_params, "lr": lr}],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(
        f"\nTraining on {dev} | {len(train_dataset)} train / {len(val_dataset)} val"
        f" | epochs={epochs} batch={batch_size} lr={lr} AMP={use_amp}"
    )
    print("=" * 65)

    best_miou = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(dev, non_blocking=True)
            masks = masks.to(dev, non_blocking=True)
            optimizer.zero_grad()

            if use_amp:
                with torch.autocast("cuda"):
                    out_ = model(images)
                    loss = criterion(out_["out"], masks)
                    if "aux" in out_:
                        loss = loss + 0.4 * criterion(out_["aux"], masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_ = model(images)
                loss = criterion(out_["out"], masks)
                if "aux" in out_:
                    loss = loss + 0.4 * criterion(out_["aux"], masks)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(len(train_loader), 1)

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.inference_mode():
            for images, masks in val_loader:
                images = images.to(dev)
                masks = masks.to(dev)
                out_ = model(images)
                val_loss += criterion(out_["out"], masks).item()
                all_preds.append(out_["out"].argmax(1).cpu().numpy())
                all_targets.append(masks.cpu().numpy())

        val_loss /= max(len(val_loader), 1)
        miou = compute_miou(
            np.concatenate(all_preds).ravel(),
            np.concatenate(all_targets).ravel(),
            num_classes,
        )
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"mIoU={miou:.4f} | {elapsed:.1f}s"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "miou": miou,
        })

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), out / "best_model.pt")
            print(f"  ✓ New best mIoU={best_miou:.4f} → {out}/best_model.pt")

    torch.save(model.state_dict(), out / "final_model.pt")
    (out / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")
    print(f"Checkpoint: {out}/best_model.pt")


# ---------------------------------------------------------------------------
# ONNX export (for Hailo Dataflow Compiler)
# ---------------------------------------------------------------------------

def export_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    img_size: tuple = (512, 512),
    device: str = "cpu",
    num_classes: int = NUM_CLASSES,
) -> None:
    """
    Export a fine-tuned DeepLabv3+ checkpoint to ONNX for Hailo compilation.

    After exporting, compile to HEF on a machine with the Hailo Dataflow Compiler:

        hailo optimize model.onnx --hw-arch hailo8l --calib-path /path/to/calib_images
        hailo compile model.har   --hw-arch hailo8l --output-dir .

    The resulting model.hef can be run on the Raspberry Pi AI Hat+ (Hailo-8L)
    via extract_silhouettes(method="hailo", hef_path="model.hef").

    Args:
        checkpoint_path: .pt file saved by train_seg()
        output_path:     destination .onnx file path
        img_size:        (H, W) — must match the training resolution (default 512x512)
        device:          "cpu" or "cuda"
        num_classes:     number of output classes (default 4)
    """
    import torch

    from gripper_cv.heapgrasp.segment import load_finetuned_model

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    model, _ = load_finetuned_model(str(checkpoint_path), device)
    model.eval()

    H, W = img_size
    dummy = torch.zeros(1, 3, H, W, device=torch.device(device))

    print(f"Exporting ONNX ({H}x{W}, {num_classes} classes)…")
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["out"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "out": {0: "batch"}},
    )

    size_mb = output_path.stat().st_size / 1e6
    print(f"Exported to {output_path}  ({size_mb:.1f} MB)")
    print()
    print("Next steps — compile for Hailo-8L AI Hat+:")
    print(f"  hailo optimize {output_path} --hw-arch hailo8l --calib-path /path/to/calib_images")
    print(f"  hailo compile model.har --hw-arch hailo8l --output-dir .")
    print()
    print("Then on the Raspberry Pi:")
    print("  gripper-cv-heapgrasp --segment hailo --hef-path model.hef")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fine-tune DeepLabv3+ for 4-class transparent-object segmentation.\n"
            "Uses synthetic data by default — no dataset download required."
        )
    )
    p.add_argument(
        "--data-dir", type=Path, default=None,
        help="Transpose-format dataset root (images/ + masks/). "
             "Omit to use SyntheticTransparentDataset.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/seg_checkpoint"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cpu",
                   help="torch device: 'cpu', 'cuda', 'mps'  (default: cpu)")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction of data for validation (default: 0.2)")
    p.add_argument("--img-size", type=int, nargs=2, default=[512, 512],
                   metavar=("H", "W"), help="Training image size (default: 512 512)")
    p.add_argument("--synthetic-length", type=int, default=1000,
                   help="Synthetic samples when --data-dir is omitted (default: 1000)")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (default: 0 — main process)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for training. Install with:\n"
            "  pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cpu"
        ) from exc

    from gripper_cv.sim2real.domain_rand import DomainRandomTransform
    from gripper_cv.training.dataset import get_dataset_classes, split_dataset

    TransposeDataset, SyntheticTransparentDataset = get_dataset_classes()
    transform = DomainRandomTransform()
    img_size = tuple(args.img_size)

    if args.data_dir is not None:
        print(f"Loading TransposeDataset from {args.data_dir}…")
        dataset = TransposeDataset(args.data_dir, transform=transform, img_size=img_size)
    else:
        print(f"Using SyntheticTransparentDataset (length={args.synthetic_length})…")
        dataset = SyntheticTransparentDataset(
            length=args.synthetic_length, img_size=img_size, transform=transform
        )

    train_ds, val_ds = split_dataset(dataset, val_fraction=args.val_split)
    train_seg(
        train_ds, val_ds,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
    )


def export_main() -> None:
    """CLI entry point for gripper-cv-export-onnx."""
    p = argparse.ArgumentParser(
        description="Export a fine-tuned DeepLabv3+ checkpoint to ONNX for Hailo compilation."
    )
    p.add_argument("--checkpoint", type=Path, required=True, metavar="PATH",
                   help="Path to .pt checkpoint saved by gripper-cv-train-seg")
    p.add_argument("--output", type=Path, default=Path("model.onnx"), metavar="PATH",
                   help="Destination .onnx file (default: model.onnx)")
    p.add_argument("--img-size", type=int, nargs=2, default=[512, 512],
                   metavar=("H", "W"), help="Model input resolution (default: 512 512)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    args = p.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        img_size=tuple(args.img_size),
        device=args.device,
        num_classes=args.num_classes,
    )


if __name__ == "__main__":
    main()
