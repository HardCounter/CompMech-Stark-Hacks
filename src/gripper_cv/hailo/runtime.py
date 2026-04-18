"""
Hailo-8L NPU inference for the Raspberry Pi AI Hat+.

HailoRT 4.x (hailo_platform) is imported lazily — this module is safe to
import on machines without the Hailo SDK installed.

Workflow to run a model on the NPU
-----------------------------------
1.  Train with gripper-cv-train-seg
2.  Export to ONNX:
        gripper-cv-export-onnx --checkpoint outputs/seg/best_model.pt --output model.onnx
3.  Compile on a machine with the Hailo Dataflow Compiler:
        hailo optimize model.onnx --hw-arch hailo8l --calib-path /calib_images
        hailo compile model.har   --hw-arch hailo8l --output-dir .
4.  Copy model.hef to the Pi and run:
        gripper-cv-heapgrasp --segment hailo --hef-path model.hef

Input convention
----------------
HailoRunner accepts CHW or NCHW float32 arrays using ImageNet normalisation
(same convention as the PyTorch pipeline).  It transposes to NHWC internally
before handing off to the Hailo runtime, and returns NCHW on the way back.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def is_hailo_available() -> bool:
    """Return True if HailoRT (hailo_platform) is installed and importable."""
    try:
        import hailo_platform  # noqa: F401
        return True
    except ImportError:
        return False


class HailoRunner:
    """
    Synchronous single-model inference on the Hailo-8L NPU (AI Hat+).

    Uses the high-level VDevice.create_infer_model API (HailoRT ≥ 4.14).
    Can be used as a context manager or standalone (call close() when done).

    Args:
        hef_path:        path to a compiled .hef model file
        batch_size:      fixed batch size (default 1)
        timeout_ms:      inference timeout in milliseconds (default 5000)

    Raises:
        ImportError       if hailo_platform is not installed
        FileNotFoundError if hef_path does not exist
    """

    def __init__(
        self,
        hef_path: str | Path,
        batch_size: int = 1,
        timeout_ms: int = 5000,
    ) -> None:
        try:
            import hailo_platform as hp
        except ImportError as exc:
            raise ImportError(
                "hailo_platform is not installed. Install the HailoRT Python SDK:\n"
                "  https://hailo.ai/developer-zone/software-downloads/"
            ) from exc

        self._hef_path = Path(hef_path)
        if not self._hef_path.exists():
            raise FileNotFoundError(f"HEF not found: {self._hef_path}")
        self._timeout_ms = timeout_ms

        # VDevice auto-discovers the Hailo-8L over PCIe
        self._vdevice = hp.VDevice()
        self._infer_model = self._vdevice.create_infer_model(str(self._hef_path))
        self._infer_model.set_batch_size(batch_size)

        # Cache stream metadata from InferModel (before configure)
        self._input_streams = self._infer_model.inputs   # list[InferModel.InferStream]
        self._output_streams = self._infer_model.outputs
        self._input_name: str = self._input_streams[0].name
        self._output_names: list[str] = [o.name for o in self._output_streams]

        # Open configured model — kept alive for the lifetime of this runner
        self._configured = self._infer_model.configure()
        self._configured.__enter__()
        self._closed = False

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def input_name(self) -> str:
        return self._input_name

    @property
    def output_names(self) -> list[str]:
        return self._output_names

    @property
    def input_shape(self) -> tuple:
        return tuple(self._input_streams[0].shape)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(self, input_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run synchronous inference on the Hailo-8L NPU.

        Args:
            input_array: float32 array, shape CHW or NCHW (PyTorch convention).
                         Values must already be ImageNet-normalised.

        Returns:
            Dict mapping each output stream name → float32 numpy array (NCHW).
        """
        arr = np.ascontiguousarray(input_array, dtype=np.float32)

        if arr.ndim == 3:
            arr = arr[np.newaxis]  # CHW → NCHW

        # Hailo expects NHWC; our callers use NCHW (PyTorch convention)
        if arr.ndim == 4:
            arr = arr.transpose(0, 2, 3, 1)  # NCHW → NHWC

        # Allocate output buffers keyed by name
        output_buffers: Dict[str, np.ndarray] = {
            o.name: np.empty(o.shape, dtype=np.float32)
            for o in self._output_streams
        }

        bindings = self._configured.create_bindings(
            input_buffers={self._input_name: arr},
            output_buffers=output_buffers,
        )

        # run() takes (list_of_bindings, timeout_ms) — both positional
        self._configured.run([bindings], self._timeout_ms)

        # Return in NCHW so callers treat it the same as PyTorch outputs.
        # Only transpose spatial maps (rank-4 tensors); leave 1-D / 2-D as-is.
        result: Dict[str, np.ndarray] = {}
        for name, buf in output_buffers.items():
            out = np.copy(buf)
            if out.ndim == 4:
                out = out.transpose(0, 3, 1, 2)  # NHWC → NCHW
            result[name] = out
        return result

    def run_single(self, input_array: np.ndarray) -> np.ndarray:
        """Convenience: runs inference and returns the first output array."""
        return self.run(input_array)[self._output_names[0]]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the configured model and VDevice."""
        if self._closed:
            return
        self._closed = True
        try:
            self._configured.__exit__(None, None, None)
        except Exception:
            pass
        try:
            self._vdevice.release()
        except Exception:
            pass

    def __enter__(self) -> "HailoRunner":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
