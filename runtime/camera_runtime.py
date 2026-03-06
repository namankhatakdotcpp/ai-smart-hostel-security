"""
runtime/camera_runtime.py — Smart Hostel AI main runtime loop (v2)
==================================================================
Integrates:
  • CentroidTracker — recognition fires once per new face, cached 2 seconds
  • SecurityService — 3 threat detectors wired to every track:
      1. REPEATED_UNKNOWN  (≥ 3 appearances in 60 s)
      2. LOITERING         (visible in-frame > 30 s)
      3. RAPID_FAIL        (≥ 5 recognition failures in 10 s)
  • Structured JSON + SQLite security event logging
  • Dwell-time and repeat alerts fire per-track (no spam)

Entry point:
    python runtime/camera_runtime.py
"""
import sys, os, cv2, time, logging, json
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.settings import (
    CAMERA_ID, TARGET_FPS, FRAME_SKIP,
    FRAME_W, FRAME_H, UNKNOWN_COOLDOWN,
    SIMILARITY_THRESHOLD, LOG_LEVEL, LOG_FILE,
)
from core.face_engine           import FaceEngine
from core.recognition           import load_all_embeddings, find_best_match, save_capture
from core.attendance            import mark_attendance
from core.tracker               import CentroidTracker
from services.telegram_service  import send_entry_alert, send_unknown_alert
from services.greeting_service  import GreetingService
from services.security_service  import SecurityService

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("runtime")

FRAME_DELAY = 1.0 / TARGET_FPS


# ── HUD helpers ────────────────────────────────────────────────────────────────

def _draw_tag(frame, bbox, label: str, color) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (10, 10, 10), 2)


def _status_bar(frame, text: str, color=(200, 200, 200)) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 32), (w, h), (22, 22, 22), -1)
    cv2.putText(frame, text, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("  Smart Hostel AI — Runtime v2 (tracker + threat engine)")
    log.info("=" * 60)

    engine   = FaceEngine()
    db       = load_all_embeddings()
    if not db:
        log.error("No embeddings found. Register students via the dashboard first.")
        return

    tracker  = CentroidTracker()
    greeter  = GreetingService()
    security = SecurityService(camera_id=CAMERA_ID)

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        log.error("Cannot open camera %d.", CAMERA_ID)
        return

    log.info("Camera %d open at %dx%d. Target %d FPS. Press 'q' to quit.",
             CAMERA_ID, FRAME_W, FRAME_H, TARGET_FPS)

    # Per-track state dictionaries (keyed by track_id)
    entered_tracks: set[int]         = set()   # track_ids already logged as entry
    unk_alert_ts:   dict[int, float] = {}      # last Telegram "unknown" ts per track
    prev_track_ids: set[int]         = set()   # to detect expired tracks

    frame_idx = 0

    while True:
        t0 = time.monotonic()

        ret, frame = cap.read()
        if not ret:
            log.warning("Frame read failed, retrying…")
            time.sleep(0.05)
            continue

        frame_idx += 1

        # ── Face detection (every FRAME_SKIP frames to save CPU) ──────────────
        raw_faces = engine.process_frame(frame) if frame_idx % FRAME_SKIP == 0 else []

        # Update tracker; returns all active Track objects
        tracks     = tracker.update(raw_faces)
        active_ids = {t.track_id for t in tracks if t.absent == 0}

        # Notify SecurityService about any tracks that just disappeared
        for tid in (prev_track_ids - active_ids):
            security.expire_track(tid)
        prev_track_ids = active_ids

        # ── Per-track processing ───────────────────────────────────────────────
        for track in tracks:
            if track.absent > 0:
                # Not visible this frame — keep drawing nothing
                continue

            # ── Recognition (only when cache is stale or track is new) ────────
            if not track.cache_valid() and track.embedding is not None:
                name, score, room = find_best_match(track.embedding, db)
                track.set_recognition(name, score, room)
                log.info("Track %d → %s (%.3f)", track.track_id, name, score)

                # ── Threat detector 3: record recognition failure ─────────────
                if name == "Unknown":
                    security.record_failure(confidence=score)

            name  = track.name
            score = track.score
            is_known = name != "Unknown"
            color    = (0, 220, 0) if is_known else (0, 0, 220)

            _draw_tag(frame, track.bbox,
                      f"{name}  {score:.2f}  #{track.track_id}", color)

            now = time.time()

            # ── Threat detector 2: loitering check (every visible frame) ──────
            img_for_loiter = save_capture(frame, name) if not is_known else None
            security.check_loitering(
                track_id   = track.track_id,
                name       = name,
                confidence = score,
                image_path = img_for_loiter,
            )

            # ── Known student: attendance + entry Telegram ─────────────────────
            if is_known:
                greeter.greet(name)
                if track.track_id not in entered_tracks:
                    if mark_attendance(name.lower(), room=track.room):
                        img_path = save_capture(frame, name)
                        send_entry_alert(name, img_path)
                        log.info("Entry logged and alert sent for '%s'.", name)
                    entered_tracks.add(track.track_id)

            # ── Unknown: repeated-detection + Telegram rate-limited alert ──────
            else:
                # Threat detector 1: repeated unknown
                security.record_unknown(
                    track_id   = track.track_id,
                    confidence = score,
                    image_path = save_capture(frame, "unknown"),
                )

                # Also rate-limit a basic Telegram unknown photo alert
                last_unk = unk_alert_ts.get(track.track_id, 0)
                if now - last_unk >= UNKNOWN_COOLDOWN:
                    img_path = save_capture(frame, "unknown")
                    send_unknown_alert(img_path)
                    unk_alert_ts[track.track_id] = now
                    log.warning("Unknown photo alert → track %d.", track.track_id)

        # ── HUD overlay ───────────────────────────────────────────────────────
        visible = sum(1 for t in tracks if t.absent == 0)
        ts_str  = datetime.now().strftime("%H:%M:%S")
        _status_bar(frame,
                    f"Faces: {visible}  |  Tracks: {tracker.active_count()}"
                    f"  |  {ts_str}  |  Press q to quit")
        cv2.putText(frame, "Smart Hostel AI", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Smart Hostel Attendance", frame)

        # FPS throttle
        elapsed = time.monotonic() - t0
        wait    = max(1, int((FRAME_DELAY - elapsed) * 1000))
        if cv2.waitKey(wait) & 0xFF == ord("q"):
            log.info("Quit command received.")
            break

    cap.release()
    cv2.destroyAllWindows()
    log.info("Runtime stopped cleanly.")


if __name__ == "__main__":
    main()
