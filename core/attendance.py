"""
core/attendance.py — SQLite attendance manager.
All SQL is contained here; NO other module should run raw queries.
"""
import sqlite3
import logging
import os
from datetime import datetime
from config.settings import DB_PATH

log = logging.getLogger(__name__)


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db() -> None:
    """Create the attendance table and unique index if they don't exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT    NOT NULL,
                date TEXT    NOT NULL,
                time TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_name_date
            ON attendance (name, date)
        """)
        conn.commit()
    log.debug("Database initialised at %s", DB_PATH)


def check_today_attendance(name: str) -> bool:
    """Return True if the student is already marked today."""
    today = datetime.now().strftime("%Y-%m-%d")
    with _connect() as conn:
        row = conn.execute(
            "SELECT id FROM attendance WHERE name = ? AND date = ?",
            (name.lower(), today),
        ).fetchone()
    return row is not None


def mark_attendance(name: str, room: str = "") -> bool:
    """
    Mark student as present for today.

    Returns:
        True  — freshly recorded.
        False — already marked today.
    """
    if check_today_attendance(name):
        log.info("'%s' already marked today.", name.title())
        return False

    today = datetime.now().strftime("%Y-%m-%d")
    now   = datetime.now().strftime("%H:%M:%S")
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)",
                (name.lower(), today, now),
            )
            conn.commit()
        log.info("✅ Attendance marked — %s | %s %s", name.title(), today, now)
        return True
    except sqlite3.IntegrityError:
        log.warning("Integrity guard — '%s' duplicate prevented.", name)
        return False


def get_today_records() -> list[dict]:
    """All attendance records for today."""
    today = datetime.now().strftime("%Y-%m-%d")
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, date, time FROM attendance WHERE date = ? ORDER BY time",
            (today,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_records() -> list[dict]:
    """Complete attendance history, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, date, time FROM attendance ORDER BY date DESC, time DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_student_records(name: str) -> int:
    """Remove all attendance records for a student. Returns rows deleted."""
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM attendance WHERE LOWER(name) = ?", (name.lower(),)
        )
        conn.commit()
    log.info("Deleted %d attendance record(s) for '%s'.", cur.rowcount, name)
    return cur.rowcount


# ── Auto-init on import ────────────────────────────────────────────────────────
initialize_db()
