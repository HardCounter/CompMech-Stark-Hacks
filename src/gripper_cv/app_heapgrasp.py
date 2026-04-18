"""
CLI entry point for the HEAPGrasp pipeline.

Usage:
    gripper-cv-heapgrasp [options]

Examples:
    # 8-view capture — auto-detects size from ArUco marker (default)
    gripper-cv-heapgrasp

    # Specify a 3 cm marker (must match the printed size exactly)
    gripper-cv-heapgrasp --marker-size 0.03

    # Disable auto-scale and set sizes manually
    gripper-cv-heapgrasp --no-auto-scale --object-diameter 0.05 --camera-distance 0.30

    # 12 views, DeepLabv3+ CNN segmentation
    gripper-cv-heapgrasp --n-views 12 --segment deeplab
"""

import argparse

from gripper_cv.heapgrasp.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HEAPGrasp: 3-D visual hull from turntable RGB views"
    )
    parser.add_argument("--n-views", type=int, default=8, help="Number of turntable views (default: 8)")
    parser.add_argument(
        "--segment",
        choices=["background", "deeplab", "finetuned", "hailo"],
        default="background",
        help="Silhouette extraction method (default: background). "
             "'hailo' runs on the Hailo-8L NPU — requires --hef-path.",
    )
    parser.add_argument("--volume", type=int, default=64, metavar="V",
                        help="Voxel grid resolution V (V³ voxels, default: 64)")
    parser.add_argument("--object-diameter", type=float, default=0.15, metavar="M",
                        help="Reconstruction volume side in metres (default: 0.15)")
    parser.add_argument("--camera-distance", type=float, default=0.40, metavar="M",
                        help="Camera distance to object centre in metres (default: 0.40)")
    parser.add_argument("--fov", type=float, default=62.2,
                        help="Camera horizontal FOV in degrees (Pi Cam v2=62.2, Module 3=66, default: 62.2)")
    parser.add_argument("--bg-threshold", type=int, default=30,
                        help="Background subtraction threshold 0-255 (default: 30)")
    parser.add_argument("--output-dir", default="outputs/heapgrasp",
                        help="Output directory (default: outputs/heapgrasp)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--device", default="cpu",
                        help="Torch device for deeplab (default: cpu)")
    parser.add_argument("--marker-size", type=float, default=0.05, metavar="M",
                        help="Physical side length of the ArUco marker in metres (default: 0.05). "
                             "Must match the printed marker size exactly.")
    parser.add_argument("--no-auto-scale", action="store_true",
                        help="Disable ArUco auto-calibration; use --object-diameter and "
                             "--camera-distance as-is.")
    parser.add_argument("--seg-checkpoint", type=str, default=None, metavar="PATH",
                        help="Path to fine-tuned .pt checkpoint (enables method='finetuned'). "
                             "Train with: gripper-cv-train-seg")
    parser.add_argument("--use-planner", action="store_true",
                        help="Run the Next-Best-View planner after reconstruction and "
                             "print recommended additional view angles.")
    parser.add_argument("--n-planner-views", type=int, default=4, metavar="N",
                        help="Number of additional views suggested by the NBV planner (default: 4).")
    parser.add_argument("--hef-path", type=str, default=None, metavar="PATH",
                        help="Path to compiled .hef model for Hailo-8L NPU (enables method='hailo'). "
                             "Compile with: gripper-cv-export-onnx → hailo optimize → hailo compile")
    parser.add_argument("--seg-img-size", type=int, nargs=2, default=[512, 512],
                        metavar=("H", "W"),
                        help="Image size the Hailo model was compiled for (default: 512 512).")

    args = parser.parse_args()
    run_pipeline(
        n_views=args.n_views,
        segment_method=args.segment,
        volume_size=args.volume,
        object_diameter=args.object_diameter,
        camera_distance=args.camera_distance,
        fov_deg=args.fov,
        bg_threshold=args.bg_threshold,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        device=args.device,
        auto_scale=not args.no_auto_scale,
        marker_size_m=args.marker_size,
        seg_checkpoint=args.seg_checkpoint,
        use_planner=args.use_planner,
        n_planner_views=args.n_planner_views,
        hef_path=args.hef_path,
        seg_img_size=tuple(args.seg_img_size),
    )


if __name__ == "__main__":
    main()
