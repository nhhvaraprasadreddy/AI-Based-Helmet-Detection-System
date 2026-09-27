# ============================================================
# violation_manager.py - Save violation evidence to DB + disk
# ============================================================
# Plate/OCR functionality has been completely removed.
# Only saves: annotated screenshot + DB record.
# ============================================================

import logging
import numpy as np

from utils.helpers  import save_violation_image
from utils.database import save_violation, log_event

logger = logging.getLogger(__name__)

# Cooldown: suppress duplicate violations within N frames per source.
# Keyed by source string.  Reset explicitly before each new video.
_last_violation_frame = {}   # type: dict
COOLDOWN_FRAMES = 30


def reset_cooldown(source):
    # type: (str) -> None
    """Reset the per-source cooldown counter.  Call before each new video."""
    _last_violation_frame.pop(source, None)


def process_violation(frame, detections, source="webcam", frame_id=0):
    # type: (np.ndarray, list, str, int) -> dict
    """
    Save evidence for a no-helmet violation.

    Steps:
      1. Cooldown check — skip if same source fired recently.
      2. Save annotated screenshot to static/evidence/violations/.
      3. Insert DB record.

    Returns challan dict or None if suppressed by cooldown.
    """
    no_helmet = [d for d in detections if d["label"] == "no_helmet"]
    if not no_helmet:
        return None

    # Per-source cooldown — uses COOLDOWN_FRAMES gap between saves
    last = _last_violation_frame.get(source, -(COOLDOWN_FRAMES + 1))
    if (frame_id - last) < COOLDOWN_FRAMES:
        return None
    _last_violation_frame[source] = frame_id

    # Highest-confidence no-helmet detection
    det  = max(no_helmet, key=lambda d: d["confidence"])
    conf = det["confidence"]

    # ── Save screenshot ───────────────────────────────────────────────
    img_url = save_violation_image(frame)

    print("===== DATABASE DEBUG =====")
    print("Violation detected")
    print("Saving image: {}".format(img_url))
    print("Inserting into database...")

    # ── Persist to DB ─────────────────────────────────────────────────
    try:
        vid = save_violation(
            violation_type = "No Helmet",
            confidence     = conf,
            image_path     = img_url,
            video_path     = None,
            source         = source,
        )
        log_event("VIOLATION", "id={} src={} conf={:.2f}".format(vid, source, conf))
        logger.info("Violation #%d saved | source=%s | conf=%.2f", vid, source, conf)
        print("Database insert successful — challan #{}".format(vid))
    except Exception as db_exc:
        logger.error("DB insert failed: %s", db_exc, exc_info=True)
        print("[ERROR] Database insert FAILED: {}".format(db_exc))
        return None

    return {
        "id":         vid,
        "confidence": conf,
        "image":      img_url,
    }
