# ============================================================
# config.py - Central configuration for the entire system
# ============================================================
# torch is imported lazily so this file is safe to import
# before torch is installed (required by install.py).
# ============================================================

import os

# ── Project root ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Standard directories ──────────────────────────────────────────────
MODELS_DIR    = os.path.join(BASE_DIR, "models")
UPLOADS_DIR   = os.path.join(BASE_DIR, "uploads")
UPLOADS_VIDEO = os.path.join(UPLOADS_DIR, "videos")
DATABASE_DIR  = os.path.join(BASE_DIR, "database")
LOGS_DIR      = os.path.join(BASE_DIR, "logs")
STATIC_DIR    = os.path.join(BASE_DIR, "static")

# ── Evidence directories — inside static/ so Flask serves natively ───
# /static/evidence/violations/  – screenshot JPEGs of each violation
# /static/evidence/videos/      – annotated output MP4s
EVIDENCE_DIR   = os.path.join(STATIC_DIR,   "evidence")
VIOLATIONS_DIR = os.path.join(EVIDENCE_DIR, "violations")
VIDEOS_DIR     = os.path.join(EVIDENCE_DIR, "videos")

# Legacy alias
OUTPUTS_DIR = EVIDENCE_DIR

# ── Database ──────────────────────────────────────────────────────────
DB_PATH = os.path.join(DATABASE_DIR, "challan.db")

# ── Model ─────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(MODELS_DIR, "yolov10n.pt")
CONFIDENCE    = 0.30          # low threshold — catches more detections
IOU_THRESHOLD = 0.45

# Lazy device detection — safe before torch is installed
def _get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

DEVICE = _get_device()

# ── COCO class indices ────────────────────────────────────────────────
PERSON_CLASS     = 0   # person
MOTORCYCLE_CLASS = 3   # motorcycle

# ── Flask ─────────────────────────────────────────────────────────────
SECRET_KEY     = "helmet_detection_secret_2024"
MAX_CONTENT_MB = 500
ALLOWED_VIDEO  = {"mp4", "avi", "mov", "mkv"}

# ── Webcam ────────────────────────────────────────────────────────────
WEBCAM_INDEX = 0
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# ── Auto-create all required directories on import ───────────────────
for _d in [MODELS_DIR, UPLOADS_VIDEO,
           VIOLATIONS_DIR, VIDEOS_DIR,
           DATABASE_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)
