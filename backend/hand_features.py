"""
Hand feature extraction shared by every inference backend.

Every backend (MediaPipe on CPU, Hailo-10H on UGen300, or the mock generator)
returns a list of `HandResult` objects holding 21 normalized keypoints.
This module turns those raw keypoints into the small set of *control signals*
the particle renderer needs:

    - openness      : 0.0 (fist) .. 1.0 (fully open hand), per hand + average
    - num_hands     : 0, 1 or 2
    - hand_distance : distance between the two palms (normalized by frame height)

All signals are smoothed with an exponential moving average (EMA) so the
particles move fluidly instead of jittering with every noisy detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# --- MediaPipe 21-keypoint hand topology -----------------------------------
WRIST = 0
MIDDLE_MCP = 9
FINGERTIPS = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky
PALM_POINTS = (0, 5, 9, 13, 17)  # used to compute the palm center

# Bone connections, used to draw the skeleton on the preview image.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
)

# Ratio of (mean fingertip-to-wrist distance) / (palm size) observed for a
# closed fist and for a fully open hand. Values in between are mapped
# linearly to 0..1. Tune these if your camera angle differs a lot.
FIST_RATIO = 1.0
OPEN_RATIO = 1.75


@dataclass
class HandResult:
    """Raw output of a backend for a single detected hand."""

    landmarks: np.ndarray  # shape (21, 3): normalized x, y in [0,1], relative z
    label: str = "Unknown"  # "Left" / "Right" from the user's point of view
    score: float = 1.0      # detection confidence


class EMA:
    """Exponential moving average: y <- y + alpha * (x - y)."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.value + self.alpha * (x - self.value)
        return self.value

    def reset(self) -> None:
        self.value = None


def to_pixels(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized landmarks to an isotropic pixel space.

    MediaPipe's x/y are normalized by image width/height separately, and z
    uses roughly the same scale as x. Scaling back to pixels makes distances
    independent of the camera aspect ratio.
    """
    return landmarks * np.array([width, height, width], dtype=np.float32)


def compute_openness(landmarks: np.ndarray, width: int, height: int) -> float:
    """How open is the hand? 0.0 = closed fist, 1.0 = fully spread.

    openness = normalize( mean(|fingertip - wrist|) / |middle_mcp - wrist| )

    Dividing by the palm size (wrist -> middle finger knuckle) makes the
    measure invariant to how far the hand is from the camera.
    """
    pts = to_pixels(landmarks, width, height)
    palm_size = np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST])
    if palm_size < 1e-6:
        return 0.0
    tip_dist = np.linalg.norm(pts[list(FINGERTIPS)] - pts[WRIST], axis=1).mean()
    ratio = tip_dist / palm_size
    return float(np.clip((ratio - FIST_RATIO) / (OPEN_RATIO - FIST_RATIO), 0.0, 1.0))


def palm_center(landmarks: np.ndarray) -> np.ndarray:
    """Normalized (x, y) center of the palm."""
    return landmarks[list(PALM_POINTS), :2].mean(axis=0)


class HandFeatureTracker:
    """Turns per-frame detections into smoothed, stable control signals."""

    def __init__(self, alpha: float = 0.35, lost_timeout: float = 0.35):
        """
        alpha        : EMA factor (higher = more responsive, lower = smoother).
        lost_timeout : seconds to keep the last hands alive after they vanish,
                       which hides single-frame detection drop-outs.
        """
        self.alpha = alpha
        self.lost_timeout = lost_timeout
        self._openness: dict[str, EMA] = {}
        self._distance = EMA(alpha)
        self._last_hands: list[dict] = []
        self._last_seen = 0.0

    def _assign_keys(self, hands: list[HandResult]) -> list[str]:
        """Stable key per hand. Falls back to left/right screen position if
        the backend reports the same handedness label twice."""
        labels = [h.label for h in hands]
        if len(set(labels)) == len(labels):
            return labels
        order = np.argsort([palm_center(h.landmarks)[0] for h in hands])
        keys = [""] * len(hands)
        for rank, idx in enumerate(order):
            keys[idx] = "Left" if rank == 0 else "Right"
        return keys

    def update(self, hands: list[HandResult], width: int, height: int, now: float) -> dict:
        """Process one frame of detections and return the control signals."""
        if not hands:
            # Hold the last known state briefly to avoid flicker.
            if self._last_hands and now - self._last_seen < self.lost_timeout:
                return self._summary(self._last_hands)
            self._openness.clear()
            self._distance.reset()
            self._last_hands = []
            return self._summary([])

        hands = sorted(hands, key=lambda h: -h.score)[:2]
        keys = self._assign_keys(hands)
        out = []
        for key, hand in zip(keys, hands):
            raw = compute_openness(hand.landmarks, width, height)
            ema = self._openness.setdefault(key, EMA(self.alpha))
            cx, cy = palm_center(hand.landmarks)
            out.append({
                "label": key,
                "openness": round(ema.update(raw), 4),
                "raw_openness": round(raw, 4),
                "center": [round(float(cx), 4), round(float(cy), 4)],
                "score": round(float(hand.score), 3),
            })

        # Drop smoothing state of hands that are no longer visible.
        for key in list(self._openness):
            if key not in keys:
                del self._openness[key]

        if len(hands) == 2:
            a = palm_center(hands[0].landmarks) * (width, height)
            b = palm_center(hands[1].landmarks) * (width, height)
            self._distance.update(float(np.linalg.norm(a - b)) / height)
        else:
            self._distance.reset()

        self._last_hands = sorted(out, key=lambda h: h["label"])
        self._last_seen = now
        return self._summary(self._last_hands)

    def _summary(self, hands: list[dict]) -> dict:
        avg = float(np.mean([h["openness"] for h in hands])) if hands else 0.0
        dist = self._distance.value if len(hands) == 2 else None
        return {
            "num_hands": len(hands),
            "hands": hands,
            "openness": round(avg, 4),
            "hand_distance": None if dist is None else round(dist, 4),
        }


def draw_skeleton(image: np.ndarray, hands: list[HandResult]) -> None:
    """Draw the 21-point skeleton of each hand onto a BGR image (in place)."""
    h, w = image.shape[:2]
    thickness = max(1, int(round(w / 320)))
    for hand in hands:
        pts = [(int(x * w), int(y * h)) for x, y, _ in hand.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(image, pts[a], pts[b], (220, 220, 220), thickness, cv2.LINE_AA)
        for i, p in enumerate(pts):
            radius = thickness + (2 if i in FINGERTIPS else 1)
            cv2.circle(image, p, radius, (255, 255, 255), -1, cv2.LINE_AA)

