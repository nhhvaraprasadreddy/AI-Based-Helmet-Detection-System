# ============================================================
# detect.py - Detection pipeline for webcam and video files
# ============================================================

import os
import cv2
import logging
import numpy as np

from utils.config  import (FRAME_WIDTH, FRAME_HEIGHT, WEBCAM_INDEX,
                            VIOLATIONS_DIR, VIDEOS_DIR, STATIC_DIR,
                            CONFIDENCE, IOU_THRESHOLD, DEVICE,
                            MODEL_PATH, PERSON_CLASS, MOTORCYCLE_CLASS)
from utils.helpers import encode_frame, _abs_to_url
from detection.helmet_detector   import HelmetDetector
from detection.violation_manager import process_violation, reset_cooldown

logger = logging.getLogger(__name__)

_detector = None   # type: HelmetDetector


def get_detector():
    # type: () -> HelmetDetector
    global _detector
    if _detector is None:
        _detector = HelmetDetector()
    return _detector


# ------------------------------------------------------------------ #
#  Live webcam / RTSP stream  (unchanged — works correctly)           #
# ------------------------------------------------------------------ #

def webcam_stream(source=WEBCAM_INDEX):
    """
    Generator yielding MJPEG frames for Flask Response streaming.
    Uses HelmetDetector which applies the rider-overlap logic.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open camera: %s", source)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    detector = get_detector()
    frame_id = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break

            detections, annotated, violation = detector.detect(frame)

            if violation:
                process_violation(annotated, detections,
                                  source=str(source), frame_id=frame_id)

            _overlay(annotated, frame_id, detector)

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + encode_frame(annotated) + b"\r\n")
            frame_id += 1
    finally:
        cap.release()
        logger.info("Camera %s released after %d frames", source, frame_id)


# ------------------------------------------------------------------ #
#  Uploaded video detection  (restored flat per-frame logic)          #
# ------------------------------------------------------------------ #

def detect_video(video_path, progress_cb=None):
    # type: (str, object) -> dict
    """
    Process an uploaded video file frame-by-frame.

    Violation condition per frame (flat boolean logic — no spatial overlap):
        person_detected AND motorcycle_detected
        AND (no_helmet_detected OR NOT helmet_detected)

    Key fixes vs previous version:
      - reset_cooldown("video") called at start so every new upload
        starts with a clean cooldown state (fixes the cross-video bug
        where frame_id resets to 0 but _last_violation_frame still holds
        the last frame number from the previous video).
      - Removed the redundant local last_saved cooldown; process_violation
        owns the single cooldown.
      - Fixed conf variable scope: each box now uses its own conf value
        when overriding the person label on violation frames.
    """
    # Reset per-source cooldown so this video starts fresh
    reset_cooldown("video")

    # ── Load model directly ───────────────────────────────────────────
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        model.to(DEVICE)
        class_names = {}
        names = model.names
        if isinstance(names, dict):
            class_names = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, list):
            class_names = {i: str(n) for i, n in enumerate(names)}
        logger.info("Video model loaded | device=%s | classes=%s",
                    DEVICE, list(class_names.values())[:10])
    except Exception as exc:
        logger.error("Failed to load model for video: %s", exc)
        return {"error": "Model load failed: {}".format(exc)}

    # ── Open video ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video: {}".format(video_path)}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── Output video writer ───────────────────────────────────────────
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(video_path))[0]
    abs_out  = os.path.join(VIDEOS_DIR, basename + "_out.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(abs_out, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        return {"error": "Cannot create output video: {}".format(abs_out)}

    violations = 0
    frame_id   = 0

    print("===== VIDEO DETECTION =====")
    print("File   : {}".format(video_path))
    print("Frames : {}  FPS: {:.1f}  Size: {}x{}".format(total, fps, w, h))
    print("Output : {}".format(abs_out))
    print("Device : {}".format(DEVICE))

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        annotated = frame.copy()

        # ── Per-frame boolean flags (reset every frame) ───────────────
        helmet_detected     = False
        no_helmet_detected  = False
        person_detected     = False
        motorcycle_detected = False

        # draw_queue stores (x1,y1,x2,y2, label, color, thickness, box_conf)
        draw_queue = []

        try:
            results = model.predict(
                frame,
                conf=CONFIDENCE,
                iou=IOU_THRESHOLD,
                device=DEVICE,
                verbose=False,
            )[0]

            detected_classes = []

            for box in results.boxes:
                cls      = int(box.cls[0])
                box_conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                name = class_names.get(cls, "cls_{}".format(cls))
                nl   = name.lower().replace(" ", "_")
                detected_classes.append("{} {:.0%}".format(name, box_conf))

                if nl in ("helmet", "with_helmet"):
                    helmet_detected = True
                    draw_queue.append((x1, y1, x2, y2,
                                       "HELMET {:.0%}".format(box_conf),
                                       (0, 200, 0), 2, box_conf))

                elif nl in ("no_helmet", "without_helmet", "no helmet"):
                    no_helmet_detected = True
                    draw_queue.append((x1, y1, x2, y2,
                                       "NO HELMET {:.0%}".format(box_conf),
                                       (0, 0, 220), 3, box_conf))

                elif nl == "person" or cls == PERSON_CLASS:
                    person_detected = True
                    draw_queue.append((x1, y1, x2, y2,
                                       "Person {:.0%}".format(box_conf),
                                       (255, 165, 0), 2, box_conf))

                elif nl == "motorcycle" or cls == MOTORCYCLE_CLASS:
                    motorcycle_detected = True
                    draw_queue.append((x1, y1, x2, y2,
                                       "Motorcycle {:.0%}".format(box_conf),
                                       (200, 0, 200), 2, box_conf))

            # ── Violation condition ───────────────────────────────────
            violation = (
                person_detected
                and motorcycle_detected
                and (no_helmet_detected or not helmet_detected)
            )

            # Debug print every 30 frames
            if frame_id % 30 == 0:
                print("===== VIDEO DEBUG =====")
                print("Frame       : {}".format(frame_id))
                print("Detected    : {}".format(detected_classes))
                print("Helmet      : {}".format(helmet_detected))
                print("No Helmet   : {}".format(no_helmet_detected))
                print("Person      : {}".format(person_detected))
                print("Motorcycle  : {}".format(motorcycle_detected))
                print("Violation   : {}".format(violation))

            # ── Draw boxes ────────────────────────────────────────────
            for x1, y1, x2, y2, label, color, thick, bc in draw_queue:
                if violation and label.startswith("Person"):
                    label = "NO HELMET - VIOLATION {:.0%}".format(bc)
                    color = (0, 0, 220)
                    thick = 3
                _draw_box(annotated, x1, y1, x2, y2, label, color, thick)

            # ── Violation banner ──────────────────────────────────────
            if violation:
                cv2.rectangle(annotated, (0, 0),
                              (annotated.shape[1], 64), (0, 0, 180), -1)
                cv2.putText(annotated,
                            "!! VIOLATION - NO HELMET DETECTED !!",
                            (10, 44), cv2.FONT_HERSHEY_DUPLEX,
                            1.0, (255, 255, 255), 2, cv2.LINE_AA)

                # Build detections list for process_violation
                # Use the highest-confidence person box_conf as the challan conf
                person_confs = [bc for _, _, _, _, lbl, _, _, bc in draw_queue
                                if lbl.startswith(("Person", "NO HELMET"))]
                challan_conf = max(person_confs) if person_confs else CONFIDENCE

                fake_det = [{"label": "no_helmet",
                             "confidence": challan_conf,
                             "box": (0, 0, w, h)}]

                result = process_violation(annotated, fake_det,
                                           source="video",
                                           frame_id=frame_id)
                if result:
                    violations += 1
                    print("  >> Challan #{} saved | frame={} conf={:.2f}".format(
                        result["id"], frame_id, challan_conf))

        except Exception as exc:
            logger.error("Frame %d error: %s", frame_id, exc, exc_info=True)
            print("[ERROR] frame {}: {}".format(frame_id, exc))

        # Frame counter overlay
        cv2.putText(annotated,
                    "Frame:{}  {}".format(frame_id, DEVICE.upper()),
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(annotated)
        frame_id += 1

        if progress_cb and frame_id % 10 == 0:
            progress_cb(min(99, int(frame_id / total * 100)))

    cap.release()
    writer.release()
    if progress_cb:
        progress_cb(100)

    output_url = _abs_to_url(abs_out)
    print("===== VIDEO DONE =====")
    print("Frames processed : {}".format(frame_id))
    print("Violations found : {}".format(violations))
    print("Output URL       : {}".format(output_url))
    logger.info("Video done: %s | frames=%d | violations=%d",
                basename, frame_id, violations)

    return {
        "output_url": output_url,
        "violations": violations,
        "frames":     frame_id,
    }


# ── Drawing helper ────────────────────────────────────────────────────

def _draw_box(frame, x1, y1, x2, y2, label, color, thickness=2):
    """Draw a filled-label bounding box on frame in-place."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
    ly = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, ly), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, max(th + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 1, cv2.LINE_AA)


# ── Webcam overlay helper ─────────────────────────────────────────────

def _overlay(frame, frame_id, detector):
    from utils.config import DEVICE
    txt = "Frame:{}  FPS:{:.0f}  {}".format(
        frame_id, detector._fps, DEVICE.upper())
    cv2.putText(frame, txt, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)
