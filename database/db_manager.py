"""
database/db_manager.py — Re-exports core/attendance.py for backwards compatibility
and provides the canonical database interface.
"""
from core.attendance import (       # noqa: F401  (re-exported)
    initialize_db,
    check_today_attendance,
    mark_attendance,
    get_today_records,
    get_all_records,
    delete_student_records,
    DB_PATH,
)

__all__ = [
    "initialize_db",
    "check_today_attendance",
    "mark_attendance",
    "get_today_records",
    "get_all_records",
    "delete_student_records",
    "DB_PATH",
]
