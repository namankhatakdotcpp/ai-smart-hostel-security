"""
core/tracker.py — Centroid-based face tracker with recognition caching.
========================================================================
How it works:
  • Each detected bounding box becomes a "Track" with a unique integer ID.
  • Between frames, tracks are matched to new detections by nearest-centroid
    distance (no external dependencies — pure NumPy).
  • Once a track is recognised (by InsightFace), the result is cached for
    CACHE_TTL seconds, so recognition only fires ONCE per new face appearance.
  • Unmatched tracks are aged out after MAX_ABSENT frames.

Performance impact:
  Recognition (the expensive InsightFace call) goes from every-frame
  to once per track → typically 5–10× fewer recognition calls → 15-25 FPS
  on CPU instead of 4-8 FPS.
"""
import time
import logging
import numpy as np
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

MAX_DISTANCE   = 120   # pixels — max centroid jump to still match same track
MAX_ABSENT     = 15    # frames — drop a track after this many misses
CACHE_TTL      = 2.0   # seconds — reuse recognition result for this long


@dataclass
class Track:
    """One tracked face across consecutive frames."""
    track_id:   int
    centroid:   np.ndarray          # (cx, cy)
    bbox:       np.ndarray          # (x1, y1, x2, y2)
    embedding:  np.ndarray | None = None
    name:       str                 = "Unknown"
    score:      float               = 0.0
    room:       str                 = ""
    absent:     int                 = 0             # frames since last detected
    recognised: bool                = False
    last_seen:  float               = field(default_factory=time.monotonic)
    recognised_at: float            = 0.0

    def cache_valid(self) -> bool:
        """True if the cached recognition result is still fresh."""
        return self.recognised and (time.monotonic() - self.recognised_at) < CACHE_TTL

    def set_recognition(self, name: str, score: float, room: str) -> None:
        self.name          = name
        self.score         = score
        self.room          = room
        self.recognised    = True
        self.recognised_at = time.monotonic()


class CentroidTracker:
    """
    Lightweight centroid tracker — no external libraries required.

    Usage in the camera loop:
        tracker = CentroidTracker()

        while True:
            faces = engine.process_frame(frame)    # only every N frames
            tracks = tracker.update(faces)

            for track in tracks:
                if not track.cache_valid():
                    name, score, room = find_best_match(track.embedding, db)
                    track.set_recognition(name, score, room)
                # draw track.name, track.score, track.bbox …
    """

    def __init__(
        self,
        max_distance: float = MAX_DISTANCE,
        max_absent:   int   = MAX_ABSENT,
    ):
        self.max_distance = max_distance
        self.max_absent   = max_absent
        self._next_id     = 0
        self._tracks: dict[int, Track] = {}

    @staticmethod
    def _centroid(bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

    def update(self, faces: list) -> list[Track]:
        """
        Match incoming face detections to existing tracks.

        Args:
            faces: List of InsightFace face objects with `.bbox` and `.embedding`.

        Returns:
            List of active Track objects (both matched and previously-cached).
        """
        now = time.monotonic()

        if not faces:
            # Age all tracks; remove stale ones
            to_delete = []
            for tid, track in self._tracks.items():
                track.absent += 1
                if track.absent > self.max_absent:
                    to_delete.append(tid)
            for tid in to_delete:
                log.debug("Track %d expired (absent %d frames).", tid, self._tracks[tid].absent)
                del self._tracks[tid]
            return list(self._tracks.values())

        # Build centroid list for new detections
        new_centroids  = np.array([self._centroid(f.bbox) for f in faces])
        new_bboxes     = [f.bbox.astype(int) for f in faces]
        new_embeddings = [f.embedding for f in faces]

        if not self._tracks:
            # No existing tracks — register all as new
            for centroid, bbox, emb in zip(new_centroids, new_bboxes, new_embeddings):
                self._register(centroid, bbox, emb, now)
            return list(self._tracks.values())

        # ── Hungarian-lite matching by nearest centroid ────────────────────────
        track_ids      = list(self._tracks.keys())
        track_cents    = np.array([self._tracks[tid].centroid for tid in track_ids])

        # Distance matrix: |tracks| × |detections|
        dists = np.linalg.norm(
            track_cents[:, None, :] - new_centroids[None, :, :], axis=2
        )  # shape (T, D)

        matched_tracks = set()
        matched_dets   = set()

        # Greedy matching: smallest distance first
        for _ in range(min(len(track_ids), len(faces))):
            t_idx, d_idx = np.unravel_index(np.argmin(dists), dists.shape)
            if dists[t_idx, d_idx] > self.max_distance:
                break
            tid = track_ids[t_idx]
            track = self._tracks[tid]
            track.centroid  = new_centroids[d_idx]
            track.bbox      = new_bboxes[d_idx]
            track.embedding = new_embeddings[d_idx]
            track.absent    = 0
            track.last_seen = now
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)
            dists[t_idx, :] = np.inf
            dists[:, d_idx] = np.inf

        # Age unmatched tracks
        to_delete = []
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self._tracks[tid].absent += 1
                if self._tracks[tid].absent > self.max_absent:
                    to_delete.append(tid)
        for tid in to_delete:
            del self._tracks[tid]

        # Register new detections that didn't match any track
        for d_idx in range(len(faces)):
            if d_idx not in matched_dets:
                self._register(new_centroids[d_idx], new_bboxes[d_idx],
                                new_embeddings[d_idx], now)

        return list(self._tracks.values())

    def _register(self, centroid, bbox, embedding, now) -> None:
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = Track(
            track_id=tid, centroid=centroid,
            bbox=bbox, embedding=embedding, last_seen=now,
        )
        log.debug("Registered new track %d at centroid %s.", tid, centroid.astype(int))

    def active_count(self) -> int:
        return len(self._tracks)
