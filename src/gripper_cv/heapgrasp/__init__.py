"""
HEAPGrasp package.

``run_pipeline`` is imported lazily so that ``import gripper_cv.heapgrasp.reconstruct``
(or other camera-free submodules) does not require ``picamera2``.  The full pipeline
still needs a Pi when you actually call ``run_pipeline``.
"""

from __future__ import annotations

from typing import Any

from .grasp import (
    GraspCandidate,
    ObjectShape,
    format_grasp_plan,
    grasp_from_voxels,
    grasps_to_json,
    pca_grasp,
    sample_grasps,
)
from .grasp_learned import LearnedGraspScorer, render_grasp_patch

__all__ = [
    "run_pipeline",
    "GraspCandidate",
    "ObjectShape",
    "pca_grasp",
    "sample_grasps",
    "grasp_from_voxels",
    "grasps_to_json",
    "format_grasp_plan",
    "LearnedGraspScorer",
    "render_grasp_patch",
]


def __getattr__(name: str) -> Any:
    if name == "run_pipeline":
        from .pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
