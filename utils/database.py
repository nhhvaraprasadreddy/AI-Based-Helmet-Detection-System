# ============================================================
# database.py - SQLite database helpers
# ============================================================

import sqlite3
import logging
from datetime import datetime
from utils.config import DB_PATH

logger = logging.getLogger(__name__)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Create the violations table if it doesn't exist, and migrate
    any existing table that was created with the old schema
    (plate_number, plate_image columns) to the new schema
    (video_path column).

    Uses ALTER TABLE ADD COLUMN which is safe to run on an already-
    migrated table because we check existing columns first.
    """
    with get_connection() as conn:
        # Create table with the full new schema
        conn.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                violation_type TEXT    NOT NULL DEFAULT 'No Helmet',
                confidence     REAL    DEFAULT 0.0,
                image_path     TEXT,
                video_path     TEXT,
                date           TEXT    NOT NULL,
                time           TEXT    NOT NULL,
                source         TEXT    DEFAULT 'webcam',
                status         TEXT    DEFAULT 'Pending'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_logs (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                event     TEXT NOT NULL,
                details   TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()

        # ── Migration: add video_path if missing ──────────────────────
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(violations)").fetchall()
        }
        print("[DB] Existing columns:", sorted(existing))

        if "video_path" not in existing:
            conn.execute("ALTER TABLE violations ADD COLUMN video_path TEXT")
            conn.commit()
            print("[DB] Migrated: added video_path column")

        # violation_type column may be missing in very old DBs
        if "violation_type" not in existing:
            conn.execute(
                "ALTER TABLE violations ADD COLUMN "
                "violation_type TEXT NOT NULL DEFAULT 'No Helmet'"
            )
            conn.commit()
            print("[DB] Migrated: added violation_type column")

    logger.info("Database ready: %s", DB_PATH)
    print("[DB] Database initialised:", DB_PATH)


def save_violation(violation_type, confidence, image_path,
                   video_path=None, source="webcam"):
    # type: (str, float, str, str, str) -> int
    """
    Insert a violation record and return its id.
    Robust to both old and new schema.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    print("===== DB INSERT DEBUG =====")
    print("violation_type : {}".format(violation_type))
    print("confidence     : {}".format(confidence))
    print("image_path     : {}".format(image_path))
    print("video_path     : {}".format(video_path))
    print("source         : {}".format(source))
    print("date/time      : {} {}".format(date_str, time_str))
    print("Inserting into database...")

    with get_connection() as conn:
        # Check which columns actually exist (handles old schema)
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(violations)").fetchall()
        }

        if "video_path" in existing and "violation_type" in existing:
            # New schema
            cur = conn.execute(
                """INSERT INTO violations
                   (violation_type, confidence, image_path, video_path,
                    date, time, source)
                   VALUES (?,?,?,?,?,?,?)""",
                (violation_type, round(confidence, 3),
                 image_path, video_path,
                 date_str, time_str, source)
            )
        elif "violation_type" in existing:
            # Partial migration — no video_path yet
            cur = conn.execute(
                """INSERT INTO violations
                   (violation_type, confidence, image_path,
                    date, time, source)
                   VALUES (?,?,?,?,?,?)""",
                (violation_type, round(confidence, 3),
                 image_path, date_str, time_str, source)
            )
        else:
            # Very old schema with plate_number
            cur = conn.execute(
                """INSERT INTO violations
                   (plate_number, violation_type, confidence, image_path,
                    date, time, source)
                   VALUES (?,?,?,?,?,?,?)""",
                ("Unknown", violation_type, round(confidence, 3),
                 image_path, date_str, time_str, source)
            )

        conn.commit()
        row_id = cur.lastrowid

    print("Database commit successful — challan #{}".format(row_id))
    logger.info("Violation #%d saved to DB", row_id)
    return row_id


def get_violation_by_id(violation_id):
    # type: (int) -> dict
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM violations WHERE id=?", (violation_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_violation(violation_id):
    # type: (int) -> None
    with get_connection() as conn:
        conn.execute("DELETE FROM violations WHERE id=?", (violation_id,))
        conn.commit()
    logger.info("Violation #%d deleted", violation_id)


def get_violations(limit=100, offset=0):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM violations ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
    return [dict(r) for r in rows]


def get_stats():
    with get_connection() as conn:
        total     = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
        today_str = datetime.now().strftime("%Y-%m-%d")
        today     = conn.execute(
            "SELECT COUNT(*) FROM violations WHERE date=?", (today_str,)
        ).fetchone()[0]
        pending   = conn.execute(
            "SELECT COUNT(*) FROM violations WHERE status='Pending'"
        ).fetchone()[0]
    return {"total": total, "today": today, "pending": pending}


def log_event(event, details=""):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO detection_logs (event,details,timestamp) VALUES (?,?,?)",
            (event, details, datetime.now().isoformat())
        )
        conn.commit()
