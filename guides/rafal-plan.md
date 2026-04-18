1. The segmentation model (silhouette extraction)

  For your turntable + fixed camera setup, background subtraction already works without any
  training. You only need a trained CNN if you want robustness to changing lighting or a
  moving camera. If you do go that route:

  - Don't train from scratch — fine-tune the pretrained DeepLabv3+ (PASCAL VOC weights,
  already in segment.py) on 50–100 photos of your actual parts with binary masks
  - Annotate with https://github.com/labelmeai/labelme (free, runs locally)
  - The parts list from the hackathon brief is your label set: spark plug, lug nut, brake pad,
   oil filter, etc.
  - Fine-tuning takes ~30 min on a laptop GPU

2. Grasp prediction — the harder and more interesting problem

  The SfS pipeline gives you a 3D point cloud. To actually decide where to grip, you'd want a
  grasp quality network. The two standard options:

  - GQ-CNN (Dex-Net) — trained on the Dex-Net 2.0 dataset (~6.7M synthetic grasps).
  Well-documented, has pretrained weights, works from depth/point cloud input.
  - GraspNet-1Billion — larger, more general, trained on 1B grasp annotations across 88
  objects.