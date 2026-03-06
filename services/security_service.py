"""
services/security_service.py — AI Security Threat Detection Engine
===================================================================
Monitors three categories of suspicious behaviour:

  1. REPEATED UNKNOWN DETECTION
     Same unknown face appears ≥ REPEAT_THRESHOLD times within REPEAT_WINDOW seconds.
     Action: Telegram alert + DB log.

  2. LOITERING DETECTION
     Any tracked face (unknown OR known) stays in view longer than LOITER_SEC.
     Action: Telegram alert + DB log.

  3. RAPID RECOGNITION FAILURES
     More than FAIL_THRESHOLD recognition failures within FAIL_WINDOW seconds
     (e.g., someone covering their face / moving erratically).
     Action: Telegram alert + DB log.

All events are persisted in SQLite (table: security_events) and optionally
written to data/threat_log.json for the dashboard.

Performance: all per-frame state is kept in Python dicts; DB writes only happen
when a threshold is crossed (not every frame).
"""

import json
import logging
import os
import sqlite3
import time
from collections import deque
from datetime import datetime

from config.settings import (
    CAMERA_ID, DB_PATH, THREAT_LOG,
    SECURITY_WINDOW, SECURITY_THRESHOLD,
)
from services.telegram_service import send_security_threat, send_unknown_alert

log = logging.getLogger(__name__)

# ── Threat thresholds ──────────────────────────────────────────────────────────
REPEAT_WINDOW    = 60    # seconds – rolling window for repeated-unknown check
REPEAT_THRESHOLD = 3     # unknown appearances within window that trigger alert
LOITER_SEC       = 30    # seconds – how long before a face is flagged as loitering
FAIL_WINDOW      = 10    # seconds – rolling window for rapid-failure check
FAIL_THRESHOLD   = 5     # failures within window that trigger suspicious behaviour


# ══════════════════════════════════════════════════════════════════════════════
# Database layer (security_events table)
# ══════════════════════════════════════════════════════════════════════════════

def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_security_table() -> None:
    """
    Create the security_events table if it does not exist.
    Called once at module import time.

    Schema:
        id          INTEGER PRIMARY KEY AUTOINCREMENT
        timestamp   TEXT    — ISO-8601 datetime string
        event_type  TEXT    — 'REPEATED_UNKNOWN' | 'LOITERING' | 'RAPID_FAIL'
        person      TEXT    — detected name or 'Unknown'
        confidence  REAL    — best cosine similarity score (0-1)
        camera_id   INTEGER — which camera triggered the event
    """
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS security_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                event_type  TEXT    NOT NULL,
                person      TEXT    NOT NULL DEFAULT 'Unknown',
                confidence  REAL    NOT NULL DEFAULT 0.0,
                camera_id   INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()
    log.debug("security_events table ready.")


def log_event_to_db(event_type: str, person: str,
                    confidence: float, camera_id: int = CAMERA_ID) -> None:
    """Insert one security event row into SQLite."""
    ts = datetime.now().isoformat()
    try:
        with _connect() as conn:
            conn.execute(
                """INSERT INTO security_events
                   (timestamp, event_type, person, confidence, camera_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (ts, event_type, person, round(float(confidence), 4), camera_id),
            )
            conn.commit()
        log.info("DB: logged %s for '%s'.", event_type, person)
    except Exception as exc:
        log.error("DB write failed for security event: %s", exc)


def get_recent_events(limit: int = 100) -> list[dict]:
    """Return the most recent `limit` security events for dashboard display."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                """SELECT * FROM security_events
                   ORDER BY timestamp DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        log.error("Failed to fetch security events: %s", exc)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Threat detection engine
# ══════════════════════════════════════════════════════════════════════════════

class SecurityService:
    """
    Stateful per-session threat detector.

    Typical integration in camera_runtime.py:
    ─────────────────────────────────────────
    security = SecurityService()

    # For each recognised face (per frame):
    if name == "Unknown":
        security.record_unknown(track_id, score, image_path)
    else:
        security.record_known_entry(track_id, name, score)

    # For recognition failures (e.g., embedding below threshold):
    security.record_failure()

    # For loitering check (call every frame that a track is visible):
    security.check_loitering(track_id, name, score, first_seen_ts, image_path)
    ─────────────────────────────────────────
    All alerting happens internally; callers don't need to manage timers.
    """

    def __init__(self, camera_id: int = CAMERA_ID):
        self.camera_id = camera_id

        # ── Detector 1: Repeated unknown ──────────────────────────────────────
        # {track_id: deque of appearance timestamps (float)}
        self._unknown_times: dict[int, deque] = {}
        # track_ids that have already triggered a REPEATED_UNKNOWN alert
        self._alerted_unknown: set[int] = set()

        # ── Detector 2: Loitering ─────────────────────────────────────────────
        # {track_id: (first_seen_ts, alert_sent)}
        self._loiter_state: dict[int, dict] = {}

        # ── Detector 3: Rapid recognition failures ────────────────────────────
        # Rolling deque of failure timestamps
        self._fail_times: deque = deque()
        self._fail_alerted_at  = 0.0   # last time we fired a rapid-fail alert

        # Last "threat bundle" alert (rolling-window unknown count)
        self._last_bundle_alert = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_unknown(
        self,
        track_id:   int,
        confidence: float = 0.0,
        image_path: str | None = None,
    ) -> bool:
        """
        Record one appearance of an unknown face (track_id).
        Returns True if a REPEATED_UNKNOWN alert was triggered.
        """
        now = time.time()

        # Initialise deque for this track
        if track_id not in self._unknown_times:
            self._unknown_times[track_id] = deque()

        q = self._unknown_times[track_id]
        q.append(now)

        # Prune timestamps older than the rolling window
        while q and (now - q[0]) > REPEAT_WINDOW:
            q.popleft()

        count = len(q)
        log.debug("Track %d unknown appearances in last %ds: %d/%d",
                  track_id, REPEAT_WINDOW, count, REPEAT_THRESHOLD)

        if count >= REPEAT_THRESHOLD and track_id not in self._alerted_unknown:
            self._alerted_unknown.add(track_id)
            self._fire_alert(
                event_type  = "REPEATED_UNKNOWN",
                person      = "Unknown",
                confidence  = confidence,
                image_path  = image_path,
                extra_msg   = (
                    f"Unknown face (track #{track_id}) detected "
                    f"{count}× in {REPEAT_WINDOW}s."
                ),
            )
            return True
        return False

    def record_failure(self, confidence: float = 0.0) -> bool:
        """
        Record one recognition failure (score below threshold).
        Returns True if a RAPID_FAIL alert was triggered.
        """
        now = time.time()
        self._fail_times.append(now)

        # Prune old failures outside the window
        while self._fail_times and (now - self._fail_times[0]) > FAIL_WINDOW:
            self._fail_times.popleft()

        count = len(self._fail_times)
        log.debug("Recognition failures in last %ds: %d/%d",
                  FAIL_WINDOW, count, FAIL_THRESHOLD)

        if count >= FAIL_THRESHOLD and (now - self._fail_alerted_at) > FAIL_WINDOW:
            self._fail_alerted_at = now
            self._fire_alert(
                event_type  = "RAPID_FAIL",
                person      = "Unknown",
                confidence  = confidence,
                image_path  = None,
                extra_msg   = (
                    f"Rapid recognition failures: {count} in {FAIL_WINDOW}s. "
                    "Possible evasion attempt."
                ),
            )
            return True
        return False

    def check_loitering(
        self,
        track_id:   int,
        name:       str,
        confidence: float,
        image_path: str | None = None,
    ) -> bool:
        """
        Call every frame that track_id is visible.
        Fires a LOITERING alert once per track when it exceeds LOITER_SEC.
        Returns True if the loitering alert fired.
        """
        now = time.time()

        if track_id not in self._loiter_state:
            self._loiter_state[track_id] = {"first_seen": now, "alerted": False}

        state    = self._loiter_state[track_id]
        duration = now - state["first_seen"]

        if duration >= LOITER_SEC and not state["alerted"]:
            state["alerted"] = True
            self._fire_alert(
                event_type  = "LOITERING",
                person      = name,
                confidence  = confidence,
                image_path  = image_path,
                extra_msg   = (
                    f"{'Unknown visitor' if name == 'Unknown' else name} "
                    f"has been in camera view for {int(duration)}s "
                    f"(track #{track_id})."
                ),
            )
            return True
        return False

    def expire_track(self, track_id: int) -> None:
        """Call when a track disappears, so its loitering timer resets."""
        self._loiter_state.pop(track_id, None)

    def recent_count(self, track_id: int) -> int:
        """Current appearance count for a track within the rolling window."""
        q = self._unknown_times.get(track_id)
        if q is None:
            return 0
        now = time.time()
        while q and (now - q[0]) > REPEAT_WINDOW:
            q.popleft()
        return len(q)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _fire_alert(
        self,
        event_type: str,
        person:     str,
        confidence: float,
        image_path: str | None,
        extra_msg:  str = "",
    ) -> None:
        """
        Centralised alert dispatcher:
          1. Log to SQLite security_events table.
          2. Append to JSON threat log (for dashboard).
          3. Send Telegram alert.
        """
        now_str   = datetime.now().strftime("%I:%M %p")
        log.critical("🚨 THREAT [%s] — %s (cam %d): %s",
                     event_type, person, self.camera_id, extra_msg)

        # 1. SQLite
        log_event_to_db(event_type, person, confidence, self.camera_id)

        # 2. JSON threat log (dashboard)
        self._append_json_log(event_type, person, now_str)

        # 3. Telegram
        label = {
            "REPEATED_UNKNOWN": "Unknown visitor detected repeatedly",
            "LOITERING":        "Loitering detected",
            "RAPID_FAIL":       "Suspicious recognition behaviour",
        }.get(event_type, event_type)

        msg = (
            f"🚨 <b>SECURITY ALERT</b>\n"
            f"{label}\n"
            f"Camera: Hostel Entrance (cam {self.camera_id})\n"
            f"Time:   {now_str}\n"
        )
        if extra_msg:
            msg += f"Detail: {extra_msg}\n"

        from services.telegram_service import _post_message  # avoid circular at module-level
        import threading
        threading.Thread(target=_post_message, args=(msg,), daemon=True).start()

        if image_path:
            send_unknown_alert(image_path)

    @staticmethod
    def _append_json_log(event_type: str, person: str, time_str: str) -> None:
        """Append a compact entry to data/threat_log.json for the dashboard."""
        log_path = THREAT_LOG
        entries  = []
        if os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    entries = json.load(f)
            except Exception:
                pass
        entries.append({
            "timestamp":  datetime.now().isoformat(),
            "time_str":   time_str,
            "event_type": event_type,
            "person":     person,
            "count":      1,
            "window_s":   REPEAT_WINDOW,
        })
        with open(log_path, "w") as f:
            json.dump(entries[-200:], f, indent=2)


def load_threat_log() -> list[dict]:
    """Read data/threat_log.json for dashboard display."""
    if not os.path.exists(THREAT_LOG):
        return []
    try:
        with open(THREAT_LOG) as f:
            return json.load(f)
    except Exception:
        return []


# ── Auto-initialise DB table on import ────────────────────────────────────────
init_security_table()
