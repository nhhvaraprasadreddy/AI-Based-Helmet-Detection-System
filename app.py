# ============================================================
# app.py - Main Flask application entry point
# ============================================================

import os
import threading
import logging

from flask import (Flask, render_template, request, jsonify,
                   Response, send_from_directory, redirect,
                   url_for, flash, abort)
from werkzeug.utils import secure_filename

from utils.helpers  import setup_logging, allowed_file
from utils.config   import (BASE_DIR, SECRET_KEY, MAX_CONTENT_MB,
                             ALLOWED_VIDEO, UPLOADS_VIDEO,
                             EVIDENCE_DIR, STATIC_DIR,
                             VIOLATIONS_DIR, VIDEOS_DIR)
from utils.database import (init_db, get_violations, get_stats,
                             delete_violation, get_violation_by_id)
from detection.detect import webcam_stream, detect_video

# ── Bootstrap ──────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)

for _d in [VIOLATIONS_DIR, VIDEOS_DIR, os.path.join(STATIC_DIR, "img")]:
    os.makedirs(_d, exist_ok=True)

init_db()

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static",
)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

_video_progress = {}   # type: dict


# ── Page routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           stats=get_stats(),
                           recent=get_violations(limit=5))


@app.route("/live")
def live():
    return render_template("live.html")


@app.route("/upload-video")
def upload_video_page():
    return render_template("upload.html")


@app.route("/history")
def history():
    page       = int(request.args.get("page", 1))
    per_page   = 20
    violations = get_violations(limit=per_page, offset=(page - 1) * per_page)
    stats      = get_stats()
    return render_template("history.html",
                           violations=violations, stats=stats, page=page)


# ── MJPEG stream ───────────────────────────────────────────────────────

@app.route("/video_feed")
def video_feed():
    source = request.args.get("src", "0")
    src    = int(source) if source.isdigit() else source
    return Response(webcam_stream(src),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Delete challan ─────────────────────────────────────────────────────

@app.route("/delete_challan/<int:violation_id>", methods=["POST"])
def delete_challan(violation_id):
    """Delete DB record and all associated evidence files."""
    try:
        violation = get_violation_by_id(violation_id)
        if violation is None:
            flash("Challan #{} not found.".format(violation_id), "warning")
            return redirect(url_for("history"))

        deleted = []
        for field in ("image_path", "video_path"):
            url = violation.get(field) or ""
            if not url:
                continue
            abs_path = _url_to_abs(url)
            if abs_path and os.path.isfile(abs_path):
                try:
                    os.remove(abs_path)
                    deleted.append(os.path.basename(abs_path))
                    logger.info("[DELETE] %s", abs_path)
                except OSError as e:
                    logger.warning("[DELETE] Could not remove %s: %s", abs_path, e)

        delete_violation(violation_id)
        msg = "Challan #{} deleted.".format(violation_id)
        if deleted:
            msg += " Files removed: {}.".format(", ".join(deleted))
        flash(msg, "success")

    except Exception as exc:
        logger.error("[DELETE] challan #%d: %s", violation_id, exc, exc_info=True)
        flash("Failed to delete challan #{}.".format(violation_id), "danger")

    return redirect(url_for("history"))


# ── Download evidence ──────────────────────────────────────────────────

@app.route("/download_evidence/<int:violation_id>")
def download_evidence(violation_id):
    """Force-download the evidence screenshot for a challan."""
    violation = get_violation_by_id(violation_id)
    if violation is None:
        flash("Challan #{} not found.".format(violation_id), "warning")
        return redirect(url_for("history"))

    url = violation.get("image_path") or ""
    if not url:
        flash("No evidence image for challan #{}.".format(violation_id), "warning")
        return redirect(url_for("history"))

    abs_path = _url_to_abs(url)
    if not abs_path or not os.path.isfile(abs_path):
        flash("Evidence file missing for challan #{}.".format(violation_id), "warning")
        return redirect(url_for("history"))

    return send_from_directory(
        os.path.dirname(abs_path),
        os.path.basename(abs_path),
        as_attachment=True,
        max_age=0,
    )


def _url_to_abs(url):
    # type: (str) -> str
    """Convert /static/... URL to absolute filesystem path."""
    if not url:
        return ""
    rel      = url.lstrip("/").replace("/", os.sep)
    abs_path = os.path.normpath(os.path.join(BASE_DIR, rel))
    if abs_path.startswith(os.path.normpath(STATIC_DIR)):
        return abs_path
    return ""


# ── Evidence serving ───────────────────────────────────────────────────

@app.route("/evidence/<path:filename>")
def serve_evidence(filename):
    abs_path = os.path.normpath(os.path.join(EVIDENCE_DIR, filename))
    if not abs_path.startswith(os.path.normpath(EVIDENCE_DIR)):
        abort(403)
    if not os.path.isfile(abs_path):
        ph = os.path.join(STATIC_DIR, "img", "no-image.png")
        if os.path.isfile(ph):
            return send_from_directory(os.path.join(STATIC_DIR, "img"), "no-image.png")
        abort(404)
    return send_from_directory(
        os.path.dirname(abs_path), os.path.basename(abs_path), max_age=0)


@app.route("/download/<path:filename>")
def download_file(filename):
    for prefix in ("/static/evidence/", "static/evidence/"):
        if filename.startswith(prefix):
            filename = filename[len(prefix):]
            break
    abs_path = os.path.normpath(os.path.join(EVIDENCE_DIR, filename))
    if not abs_path.startswith(os.path.normpath(EVIDENCE_DIR)):
        abort(403)
    if not os.path.isfile(abs_path):
        abort(404)
    return send_from_directory(
        os.path.dirname(abs_path), os.path.basename(abs_path),
        as_attachment=True, max_age=0)


# ── API endpoints ──────────────────────────────────────────────────────

@app.route("/api/upload-video", methods=["POST"])
def api_upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if not allowed_file(f.filename, ALLOWED_VIDEO):
        return jsonify({"error": "Invalid file type"}), 400
    filename = secure_filename(f.filename)
    path     = os.path.join(UPLOADS_VIDEO, filename)
    f.save(path)
    _video_progress[filename] = 0

    def _run():
        def _cb(pct):
            _video_progress[filename] = pct
        result = detect_video(path, progress_cb=_cb)
        _video_progress[filename] = 100
        logger.info("Video done: %s violations=%s",
                    filename, result.get("violations"))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "processing", "filename": filename})


@app.route("/api/video-progress/<filename>")
def api_video_progress(filename):
    return jsonify({"progress": _video_progress.get(filename, -1)})


@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())


@app.route("/api/violations")
def api_violations():
    limit  = int(request.args.get("limit", 20))
    offset = int(request.args.get("offset", 0))
    return jsonify(get_violations(limit=limit, offset=offset))


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  Helmet Detection System")
    logger.info("  BASE_DIR     : %s", BASE_DIR)
    logger.info("  EVIDENCE_DIR : %s", EVIDENCE_DIR)
    logger.info("  http://127.0.0.1:5000")
    logger.info("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
