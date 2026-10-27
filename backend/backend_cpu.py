"""
CPU backend: Google MediaPipe Hands.

This is the backend used in the live demo. It runs palm detection + the
21-keypoint hand landmark model entirely on the CPU (~30 FPS on a laptop).
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from hand_features import HandResult


class MediaPipeBackend:
    name = "cpu"
    label = "CPU (MediaPipe)"
    needs_camera = True

    def __init__(self, max_hands: int = 2, model_complexity: int = 0,
                 min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.5):
        # model_complexity=0 is the "lite" landmark model: lower latency,
        # and the same network family we target on the Hailo-10H.
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> list[HandResult]:
        """Detect hands in a (mirrored) BGR frame."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # lets MediaPipe avoid a copy
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return []

        hands = []
        for landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            pts = np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32)
            cls = handedness.classification[0]
            # The frame is already mirrored (selfie view), which is what
            # MediaPipe assumes, so the label matches the user's real hand.
            hands.append(HandResult(pts, cls.label, float(cls.score)))
        return hands

    def close(self) -> None:
        self._hands.close()
