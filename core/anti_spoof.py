"""
core/anti_spoof.py — Blink-based liveness detector using MediaPipe Face Mesh.
"""
import cv2
import time
import logging
import numpy as np
import mediapipe as mp
from scipy.spatial import distance

log = logging.getLogger(__name__)

EAR_THRESHOLD     = 0.22
CLOSED_FRAMES_MIN = 2
LIVENESS_TIMEOUT  = 3.0

LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _ear(landmarks, indices: list[int], w: int, h: int) -> float:
    pts = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in indices],
        dtype=np.float32,
    )
    a = distance.euclidean(pts[1], pts[5])
    b = distance.euclidean(pts[2], pts[4])
    c = distance.euclidean(pts[0], pts[3])
    return (a + b) / (2.0 * c) if c > 0 else 0.0


class AntiSpoof:
    """
    Session-level liveness detector (blink detection via EAR algorithm).

    Call reset() when a new face appears, then check(frame) each frame:
        True  → live, allow recognition
        False → timeout / rejected
        None  → still waiting for blink
    """

    def __init__(
        self,
        ear_threshold: float = EAR_THRESHOLD,
        closed_frames_min: int = CLOSED_FRAMES_MIN,
        liveness_timeout: float = LIVENESS_TIMEOUT,
    ):
        self.ear_threshold     = ear_threshold
        self.closed_frames_min = closed_frames_min
        self.liveness_timeout  = liveness_timeout
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self._reset_state()

    def _reset_state(self):
        self._session_start  = time.monotonic()
        self._closed_counter = 0
        self._is_alive       = False
        self._timed_out      = False

    def reset(self):
        """Start a fresh liveness session."""
        self._reset_state()

    def check(self, frame_bgr: np.ndarray) -> bool | None:
        if self._is_alive:  return True
        if self._timed_out: return False

        elapsed = time.monotonic() - self._session_start
        if elapsed > self.liveness_timeout:
            self._timed_out = True
            log.warning("Anti-spoof timeout — no blink detected.")
            return False

        rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        h, w, _ = frame_bgr.shape
        lm       = results.multi_face_landmarks[0].landmark
        avg_ear  = (_ear(lm, LEFT_EYE, w, h) + _ear(lm, RIGHT_EYE, w, h)) / 2.0

        if avg_ear < self.ear_threshold:
            self._closed_counter += 1
        else:
            if self._closed_counter >= self.closed_frames_min:
                self._is_alive = True
                log.info("Blink confirmed (%d frames). Person is live.", self._closed_counter)
                return True
            self._closed_counter = 0

        return None

    def time_remaining(self) -> float:
        return max(0.0, self.liveness_timeout - (time.monotonic() - self._session_start))
