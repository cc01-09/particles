"""
UGen300 backend: Hailo-10H NPU via HailoRT (`hailo_platform`).

!!! INTERFACE PROTOTYPE - NOT TESTED ON REAL HARDWARE !!!

This file shows how the hand landmark model is meant to be loaded and run
on the ASUS UGen300 (Hailo-10H) with the HailoRT asynchronous inference API
(VDevice -> create_infer_model -> configure -> bindings -> run).
It was written without physical hardware, so:

  1. Palm detection is NOT implemented. `hand_landmark_lite` expects a
     hand-centered crop. A real pipeline must first run a palm detector
     (e.g. `palm_detection_lite.hef`), decode its anchors, apply NMS and
     compute a rotated ROI. Here a fixed center crop is used as a stand-in.
  2. Output post-processing is a best-effort guess. Output tensor names,
     order, quantization and coordinate ranges depend on how the HEF was
     compiled. Check them with `hailortcli parse-hef hand_landmark_lite.hef`
     and adapt `_decode()` accordingly.

Everything downstream (openness, smoothing, WebSocket, renderer) is
backend-agnostic, so once `process()` returns correct `HandResult`s the
rest of the system works unchanged.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from hand_features import HandResult

try:
    from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice
except ImportError as exc:  # pragma: no cover - only on machines without HailoRT
    raise RuntimeError(
        "hailo_platform is not installed. Install the HailoRT Python wheel that "
        "matches your UGen300 driver/firmware (see README > UGen300 deployment)."
    ) from exc


class HailoHandBackend:
    name = "hailo"
    label = "UGen300 (Hailo-10H)"
    needs_camera = True

    def __init__(self, hef_path: str = "models/hand_landmark_lite.hef",
                 timeout_ms: int = 1000, presence_threshold: float = 0.5):
        hef = Path(hef_path)
        if not hef.exists():
            raise FileNotFoundError(
                f"HEF not found: {hef}. Download/compile hand_landmark_lite.hef "
                "for Hailo-10H and pass it with --hef."
            )
        self.timeout_ms = timeout_ms
        self.presence_threshold = presence_threshold

        # ExitStack keeps the VDevice and configured model open for the
        # lifetime of the backend (they are context managers in HailoRT).
        self._stack = ExitStack()

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self._vdevice = self._stack.enter_context(VDevice(params))

        # Load the compiled network (.hef) onto the NPU.
        self._infer_model = self._vdevice.create_infer_model(str(hef))
        self._infer_model.set_batch_size(1)
        self._infer_model.input().set_format_type(FormatType.UINT8)
        for output in self._infer_model.outputs:
            # Ask HailoRT to de-quantize outputs to float32 for us.
            output.set_format_type(FormatType.FLOAT32)

        self._configured = self._stack.enter_context(self._infer_model.configure())

        # Model input is NHWC without the batch dim, e.g. (224, 224, 3).
        self._in_h, self._in_w = self._infer_model.input().shape[:2]
        self._output_names = list(self._infer_model.output_names)
        self._output_buffers = {
            name: np.empty(self._infer_model.output(name).shape, dtype=np.float32)
            for name in self._output_names
        }
        self._bindings = self._configured.create_bindings(output_buffers=self._output_buffers)

    # ------------------------------------------------------------------ #
    # Pipeline: ROI selection -> preprocess -> NPU inference -> decode
    # ------------------------------------------------------------------ #
    def process(self, frame_bgr: np.ndarray) -> list[HandResult]:
        roi = self._select_roi(frame_bgr)
        tensor = self._preprocess(frame_bgr, roi)

        self._bindings.input().set_buffer(tensor)
        self._configured.run([self._bindings], self.timeout_ms)
        outputs = {name: self._bindings.output(name).get_buffer() for name in self._output_names}

        hand = self._decode(outputs, roi, frame_bgr.shape)
        return [hand] if hand is not None else []

    def _select_roi(self, frame: np.ndarray) -> tuple[int, int, int]:
        """Return a square ROI (x0, y0, size) in pixels.

        TODO(real hardware): replace with palm detection (palm_detection_lite.hef)
        + anchor decoding + NMS + rotated ROI, as in MediaPipe's pipeline.
        Placeholder: a fixed center square covering 70% of the frame height.
        """
        h, w = frame.shape[:2]
        size = int(min(h, w) * 0.7)
        return (w - size) // 2, (h - size) // 2, size

    def _preprocess(self, frame: np.ndarray, roi: tuple[int, int, int]) -> np.ndarray:
        x0, y0, size = roi
        crop = frame[y0:y0 + size, x0:x0 + size]
        crop = cv2.resize(crop, (self._in_w, self._in_h), interpolation=cv2.INTER_LINEAR)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(crop, dtype=np.uint8)

    def _decode(self, outputs: dict[str, np.ndarray], roi: tuple[int, int, int],
                frame_shape: tuple[int, ...]) -> HandResult | None:
        """Convert raw output tensors into a HandResult.

        TODO(real hardware): verify against `hailortcli parse-hef`.
        Assumed outputs of hand_landmark_lite (MediaPipe-compatible):
          - 63 floats : 21 x (x, y, z) landmarks in input-pixel units
          - 1 float   : hand presence score (logit or probability)
          - 1 float   : handedness score (> 0.5 = right hand)
          - 63 floats : world landmarks in meters (unused here)
        """
        flat = {name: np.asarray(buf, dtype=np.float32).reshape(-1) for name, buf in outputs.items()}
        landmark_outs = [v for v in flat.values() if v.size == 63]
        scalar_outs = [float(v[0]) for v in flat.values() if v.size == 1]
        if not landmark_outs:
            return None

        presence = _as_probability(scalar_outs[0]) if scalar_outs else 1.0
        if presence < self.presence_threshold:
            return None
        handedness = _as_probability(scalar_outs[1]) if len(scalar_outs) > 1 else 0.5

        # Map from crop pixel space back to normalized full-frame coordinates.
        x0, y0, size = roi
        h, w = frame_shape[:2]
        lm = landmark_outs[0].reshape(21, 3).copy()
        lm[:, 0] = (x0 + lm[:, 0] / self._in_w * size) / w
        lm[:, 1] = (y0 + lm[:, 1] / self._in_h * size) / h
        lm[:, 2] = lm[:, 2] / self._in_w * size / w

        label = "Right" if handedness > 0.5 else "Left"
        return HandResult(lm, label, presence)

    def close(self) -> None:
        self._stack.close()


def _as_probability(x: float) -> float:
    """Apply a sigmoid if the value looks like a logit rather than a probability."""
    return x if 0.0 <= x <= 1.0 else float(1.0 / (1.0 + np.exp(-x)))
