# ============================================================
# helpers.py - Shared utility functions
# ============================================================

import os
import cv2
import shutil
import logging
import tempfile
import numpy as np
from datetime import datetime
from utils.config import VIOLATIONS_DIR, LOGS_DIR, STATIC_DIR

logger = logging.getLogger(__name__)


# ── Safe image write ──────────────────────────────────────────────────
# cv2.imwrite silently returns False on Windows paths with spaces
# (OneDrive, Desktop).  Write to a temp file first, then move.

def _safe_imwrite(abs_path, frame):
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    try:
        ok = cv2.imwrite(abs_path, frame)
        if ok and os.path.exists(abs_path) and os.path.getsize(abs_path) > 0:
            return True
    except Exception:
        pass
    try:
        suffix = os.path.splitext(abs_path)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tmp = tf.name
        ok = cv2.imwrite(tmp, frame)
        if ok and os.path.exists(tmp):
            shutil.move(tmp, abs_path)
            return os.path.exists(abs_path)
    except Exception as exc:
        print("[ERROR] _safe_imwrite: {}".format(exc))
    return False


# ── Path ↔ URL conversion ─────────────────────────────────────────────

def _abs_to_url(abs_path):
    # type: (str) -> str
    """Absolute path inside static/  →  /static/evidence/... URL."""
    rel = os.path.relpath(
        os.path.normpath(abs_path),
        os.path.normpath(STATIC_DIR)
    )
    return "/static/" + rel.replace("\\", "/")


# ── Evidence saving ───────────────────────────────────────────────────

def save_violation_image(frame, prefix="violation"):
    # type: (np.ndarray, str) -> str
    """
    Save annotated violation screenshot to static/evidence/violations/.
    Returns browser URL or empty string on failure.
    """
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = "{}_{}.jpg".format(prefix, ts)
    abs_path = os.path.join(VIOLATIONS_DIR, filename)

    ok = _safe_imwrite(abs_path, frame)
    print("[DEBUG] Evidence path  : {}".format(abs_path))
    print("[DEBUG] Evidence exists: {}".format(os.path.exists(abs_path)))

    if not ok or not os.path.exists(abs_path):
        print("[ERROR] Evidence NOT saved: {}".format(abs_path))
        return ""

    url = _abs_to_url(abs_path)
    print("[DEBUG] Evidence URL   : {}".format(url))
    logger.info("Evidence saved: %s", url)
    return url


# ── Frame encoding ────────────────────────────────────────────────────

def encode_frame(frame):
    # type: (np.ndarray) -> bytes
    """JPEG-encode a frame for MJPEG streaming."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


# ── Misc ──────────────────────────────────────────────────────────────

def allowed_file(filename, allowed_set):
    # type: (str, set) -> bool
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


def setup_logging():
    """Configure root logger — guarantees logs dir exists first."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, "app.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
