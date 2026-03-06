"""
core/recognition.py — Multi-embedding face recognition + student registration.

This is the single module that owns:
  • Loading / saving student embeddings
  • Cosine-similarity best-match recognition
  • Frame capture to disk
  • Student registration (20-sample live webcam flow)
  • Admin helpers: list / delete students
"""
import cv2
import glob
import logging
import os
import pickle
import numpy as np
from datetime import datetime

from config.settings import (
    EMBEDDINGS_DIR, CAPTURES_DIR, SIMILARITY_THRESHOLD,
    FRAMES_TO_CAPTURE, BLUR_THRESHOLD, CAMERA_ID,
)

log = logging.getLogger(__name__)

# Alias so dashboard can import SIMILARITY_THRESHOLD directly from this module
SIMILARITY_THRESHOLD = SIMILARITY_THRESHOLD   # re-exported


# ══════════════════════════════════════════════════════════════════════════════
# Embedding I/O
# ══════════════════════════════════════════════════════════════════════════════

def _pkl_path(name: str) -> str:
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    return os.path.join(EMBEDDINGS_DIR, f"{name.lower().replace(' ', '_')}.pkl")


def _get_embs(data: dict | list) -> list:
    """
    Extract the embedding list from a pkl payload.
    Handles both new format (key='embeddings') and old format (key='embedding').
    Also handles a single ndarray stored directly as the old single-embedding format.
    """
    if isinstance(data, dict):
        # New multi-embed format
        if "embeddings" in data and data["embeddings"]:
            return list(data["embeddings"])
        # Old single-embed format stored as list under 'embedding' key
        if "embedding" in data:
            val = data["embedding"]
            if isinstance(val, np.ndarray):
                return [val]   # wrap single embedding in a list
            if isinstance(val, list) and len(val) > 0:
                return val
        return []
    # Raw list / ndarray at top level (legacy shared pkl)
    if isinstance(data, (list, np.ndarray)):
        return list(data) if len(data) > 0 else []
    return []


def load_all_embeddings() -> dict:
    """
    Load every per-student .pkl file from EMBEDDINGS_DIR.
    Handles old single-embedding format and new multi-embedding format.

    Returns:
        {
            "naman": {"room": "A-203", "embeddings": [emb1, …, emb20]},
            …
        }
    """
    db: dict = {}
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    for fpath in glob.glob(os.path.join(EMBEDDINGS_DIR, "*.pkl")):
        try:
            with open(fpath, "rb") as f:
                data = pickle.load(f)
            embs = _get_embs(data)
            if embs:
                key = (data.get("name") if isinstance(data, dict) else None) \
                      or os.path.basename(fpath).replace(".pkl", "")
                db[key.lower()] = {
                    "room":       data.get("room", "") if isinstance(data, dict) else "",
                    "embeddings": embs,
                }
                log.debug("Loaded %d embedding(s) for '%s'.", len(embs), key)
            else:
                log.warning("No embeddings found in %s — re-register this student.", fpath)
        except Exception as exc:
            log.warning("Could not load %s: %s", fpath, exc)

    log.info("Loaded %d student(s): %s", len(db), list(db.keys()))
    return db


def save_student(name: str, room: str, embeddings: list) -> str:
    """Persist a student's embedding list to data/embeddings/<name>.pkl."""
    path = _pkl_path(name)
    with open(path, "wb") as f:
        pickle.dump({"name": name.lower(), "room": room, "embeddings": embeddings}, f)
    log.info("Saved %d embeddings for '%s' → %s", len(embeddings), name, path)
    return path


def delete_student(name: str) -> bool:
    """Remove a student's embedding file. Returns True if it existed."""
    path = _pkl_path(name)
    if os.path.exists(path):
        os.remove(path)
        log.info("Deleted embeddings for '%s'.", name)
        return True
    return False


def list_all_students() -> list[dict]:
    """Return [{name, room, embedding_count}] for all registered students."""
    out = []
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    for fname in sorted(os.listdir(EMBEDDINGS_DIR)):
        if not fname.endswith(".pkl"):
            continue
        try:
            with open(os.path.join(EMBEDDINGS_DIR, fname), "rb") as f:
                data = pickle.load(f)
            embs = _get_embs(data)
            out.append({
                "name":            (data.get("name") if isinstance(data, dict) else fname.replace(".pkl", "")).title(),
                "room":            data.get("room", "—") if isinstance(data, dict) else "—",
                "embedding_count": len(embs),
            })
        except Exception as exc:
            log.warning("Skipping %s: %s", fname, exc)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Similarity & Best-match
# ══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if (na and nb) else 0.0


def find_best_match(query: np.ndarray, db: dict) -> tuple[str, float, str]:
    """
    Compare query embedding against ALL stored embeddings per student and
    return the best (maximum cosine similarity) match.

    Returns:
        (name, score, room) — name is 'Unknown' when score < SIMILARITY_THRESHOLD.
    """
    best_name, best_score, best_room = "Unknown", -1.0, ""
    for key, student in db.items():
        for emb in student["embeddings"]:
            score = cosine_similarity(query, emb)
            if score > best_score:
                best_score, best_name, best_room = score, key, student["room"]

    if best_score >= SIMILARITY_THRESHOLD:
        return best_name.title(), best_score, best_room
    return "Unknown", best_score, ""


# ══════════════════════════════════════════════════════════════════════════════
# Frame capture
# ══════════════════════════════════════════════════════════════════════════════

def save_capture(frame: np.ndarray, name: str) -> str:
    """
    Save a BGR frame to captures/<name>_<timestamp>.jpg.
    Returns the file path.
    """
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe = name.lower().replace(" ", "_")
    path = os.path.join(CAPTURES_DIR, f"{safe}_{ts}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    log.info("Capture saved: %s", path)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Student registration (live webcam, 20 samples)
# ══════════════════════════════════════════════════════════════════════════════

def _is_blurry(frame: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD


def _is_centred(bbox, w, h, tol=0.30) -> bool:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return abs(cx - w / 2) < w * tol and abs(cy - h / 2) < h * tol


def register_student(
    student_name: str,
    room: str = "",
    frames_to_capture: int = FRAMES_TO_CAPTURE,
    camera_id: int = CAMERA_ID,
) -> bool:
    """
    Open the webcam and capture `frames_to_capture` valid face embeddings.
    Saves embeddings to data/embeddings/<name>.pkl on success.

    Returns True on success, False on failure / abort.
    """
    from core.face_engine import FaceEngine  # local import to avoid circular

    key = student_name.strip().lower()
    if not key:
        log.error("Empty student name — aborting registration.")
        return False

    log.info("Starting registration for '%s' (room=%s), target=%d frames.",
             student_name, room or "N/A", frames_to_capture)

    engine = FaceEngine()
    cap    = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        log.error("Cannot open camera %d.", camera_id)
        return False

    captured: list[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        display  = frame.copy()

        if _is_blurry(frame):
            status, color = "Blurry — hold still!", (0, 140, 255)
        else:
            faces = engine.process_frame(frame)
            if not faces:
                status, color = "No face detected", (0, 0, 220)
            elif len(faces) > 1:
                status, color = "Multiple faces — one person only", (0, 0, 220)
            else:
                face = faces[0]
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                if not _is_centred(bbox, w, h):
                    status, color = "Move face to centre", (0, 140, 255)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                else:
                    captured.append(face.embedding)
                    n  = len(captured)
                    status, color = f"Captured {n}/{frames_to_capture}", (0, 220, 0)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    log.debug("Frame %d captured.", n)

        # Overlay UI
        bar_w    = w - 40
        progress = int(bar_w * len(captured) / frames_to_capture)
        cv2.rectangle(display, (20, h - 50), (20 + bar_w, h - 30), (60, 60, 60), -1)
        cv2.rectangle(display, (20, h - 50), (20 + progress,  h - 30), (0, 200, 80), -1)
        cv2.putText(display, status,  (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        cv2.putText(display, f"{student_name} | Room: {room or 'N/A'}",
                    (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.rectangle(display, (int(w*.28), int(h*.12)),
                               (int(w*.72), int(h*.88)), (100, 100, 100), 1)
        cv2.imshow("Student Registration", display)

        if len(captured) >= frames_to_capture:
            log.info("All %d frames captured.", frames_to_capture)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            log.info("Registration aborted by user.")
            cap.release()
            cv2.destroyAllWindows()
            return False

    cap.release()
    cv2.destroyAllWindows()

    if captured:
        save_student(student_name, room, captured)
        log.info("Registration complete — %d embeddings for '%s'.", len(captured), student_name)
        return True

    log.error("No embeddings captured — registration failed.")
    return False
