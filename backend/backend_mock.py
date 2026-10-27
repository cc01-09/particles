"""
Mock backend: synthetic hands, no camera required.

Generates anatomically plausible 21-point hands that open, close and move
around the frame over time. Useful for testing the full pipeline (feature
extraction, WebSocket, renderer) on machines without a webcam, or for
rehearsing a demo.

Timeline (repeats every 30 s):
   0-12 s  one hand sweeping left <-> right while opening / closing
  12-24 s  two hands, each drifting on its own side
  24-30 s  no hands (idle)
"""

from __future__ import annotations

import math
import time

import numpy as np

from hand_features import HandResult

# Each finger: (first joint index, base point, direction, segment lengths).
# Units are "palm lengths" (wrist -> middle knuckle = 1), y points up (negative).
_FINGERS = (
    (1, (-0.30, -0.25), (-0.75, -0.66), (0.0, 0.35, 0.30, 0.25)),   # thumb 1-4
    (5, (-0.30, -0.95), (-0.10, -1.00), (0.0, 0.45, 0.27, 0.22)),   # index 5-8
    (9, (0.00, -1.00), (0.00, -1.00), (0.0, 0.50, 0.30, 0.24)),     # middle 9-12
    (13, (0.25, -0.95), (0.10, -1.00), (0.0, 0.45, 0.28, 0.22)),    # ring 13-16
    (17, (0.48, -0.85), (0.25, -1.00), (0.0, 0.35, 0.22, 0.20)),    # pinky 17-20
)
_CURL_ANGLES = (0.0, 1.45, 1.75, 1.25)  # radians per joint when fully curled


def synthetic_hand(curl: float) -> np.ndarray:
    """21x3 hand in palm-length units. curl: 0 = open, 1 = fist."""
    pts = np.zeros((21, 3), dtype=np.float32)
    for first, base, direction, lengths in _FINGERS:
        d = np.array(direction, dtype=np.float32)
        d /= np.linalg.norm(d)
        is_thumb = first == 1
        pos = np.array([*base, 0.0], dtype=np.float32)
        angle = 0.0
        for j, seg in enumerate(lengths):
            angle += _CURL_ANGLES[j] * curl * (0.55 if is_thumb else 1.0)
            # Rotate the in-plane direction towards the camera (-z) as it curls.
            step = np.array([d[0] * math.cos(angle), d[1] * math.cos(angle), -math.sin(angle)])
            pos = pos + step * seg
            pts[first + j] = pos
    return pts


class MockBackend:
    name = "mock"
    label = "Mock (synthetic)"
    needs_camera = False

    def __init__(self):
        self._t0 = time.perf_counter()

    def process(self, frame_bgr: np.ndarray) -> list[HandResult]:
        h, w = frame_bgr.shape[:2]
        t = time.perf_counter() - self._t0
        phase = t % 30.0

        if phase < 12:      # one hand sweeping across the frame
            curl = 0.5 + 0.5 * math.sin(t * 0.8)
            cx = 0.5 + 0.3 * math.sin(t * 0.5)
            cy = 0.55 + 0.08 * math.sin(t * 0.9)
            return [self._place(synthetic_hand(curl), cx, cy, 0.16, w, h, mirror=True, label="Right")]
        if phase < 24:      # two hands on either side
            hands = []
            for i, side in enumerate((-1, 1)):
                curl = 0.5 + 0.5 * math.sin(t * 0.9 + i * 1.6)
                cx = 0.5 + side * (0.24 + 0.08 * math.sin(t * 0.4 + i))
                cy = 0.55 + 0.1 * math.sin(t * 0.6 + i * 2.0)
                hands.append(self._place(synthetic_hand(curl), cx, cy, 0.16, w, h, mirror=side > 0,
                                         label="Left" if side < 0 else "Right"))
            return hands
        return []

    @staticmethod
    def _place(local: np.ndarray, cx: float, cy: float, scale: float, w: int, h: int,
               mirror: bool, label: str) -> HandResult:
        """Project palm-unit coordinates into normalized image coordinates."""
        aspect = h / w
        sx = -1.0 if mirror else 1.0
        lm = np.empty_like(local)
        lm[:, 0] = cx + sx * local[:, 0] * scale * aspect
        lm[:, 1] = cy + local[:, 1] * scale
        lm[:, 2] = local[:, 2] * scale * aspect
        return HandResult(lm, label, 0.99)

    def close(self) -> None:
        pass
