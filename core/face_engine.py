"""
core/face_engine.py — InsightFace ArcFace detection and embedding extraction.
"""
import logging
import numpy as np
from insightface.app import FaceAnalysis
from config.settings import LOG_LEVEL

log = logging.getLogger(__name__)


class FaceEngine:
    """
    Detects faces and extracts 512-D ArcFace embeddings via InsightFace.
    Forced to CPUExecutionProvider for Mac/CPU compatibility.
    """

    def __init__(self, model_name: str = "buffalo_l"):
        log.info("Initialising FaceEngine (model=%s)…", model_name)
        self.app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        log.info("FaceEngine ready.")

    def process_frame(self, frame: np.ndarray) -> list:
        """
        Detect all faces in a BGR OpenCV frame.

        Returns:
            List of face objects; each has `.bbox` (np.ndarray) and
            `.embedding` (np.ndarray of shape (512,)).
        """
        return self.app.get(frame)
