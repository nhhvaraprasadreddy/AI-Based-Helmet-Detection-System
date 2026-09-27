# ============================================================
# helmet_detector.py  –  YOLOv10 + smart rider violation logic
# ============================================================
#
# VIOLATION LOGIC
# ---------------
# A violation is raised ONLY when ALL three conditions are true:
#   1. A motorcycle is detected in the frame.
#   2. A person bounding box OVERLAPS or is NEAR that motorcycle
#      (the person is the rider, not a random pedestrian).
#   3. The head region of that person has NO helmet.
#
# Pedestrians (persons far from any motorcycle) are labelled but
# do NOT trigger a violation or generate a challan.
#
# HELMET CHECK
# ------------
# Uses a two-factor HSV + Canny heuristic on the top 25 % of the
# person bounding box (the head region).
# ============================================================

import os
import time
import cv2
import logging
import numpy as np

from utils.config import (MODEL_PATH, CONFIDENCE, IOU_THRESHOLD,
                           DEVICE, PERSON_CLASS, MOTORCYCLE_CLASS)

logger = logging.getLogger(__name__)

# ── Colours (BGR) ─────────────────────────────────────────────────────
COLOR_HELMET      = (0,   200,   0)   # green  – helmeted rider
COLOR_NO_HELMET   = (0,     0, 220)   # red    – violation rider
COLOR_PEDESTRIAN  = (255, 165,   0)   # orange – person not on bike
COLOR_MOTORCYCLE  = (200,   0, 200)   # purple

# ── Heuristic tuning ──────────────────────────────────────────────────
HEAD_FRAC          = 0.25   # top 25 % of person box = head region
HELMET_PIXEL_RATIO = 0.14   # min helmet-like pixel fraction
EDGE_DENSITY_MIN   = 0.030  # min Canny edge density

# Rider–motorcycle association: a person is considered a rider when
# the vertical overlap between the person box and the motorcycle box
# is at least this fraction of the person's height.
RIDER_OVERLAP_FRAC = 0.20   # 20 % vertical overlap required

# HSV ranges for helmet-like pixels
_DARK_LO  = np.array([  0,   0,  10], dtype=np.uint8)   # black / dark grey
_DARK_HI  = np.array([180,  65, 185], dtype=np.uint8)
_COLOR_LO = np.array([  0,  65,  65], dtype=np.uint8)   # vivid colours
_COLOR_HI = np.array([180, 255, 255], dtype=np.uint8)


class HelmetDetector:
    """
    Wraps a YOLOv10 model.

    Supports two modes (auto-detected at load time):
      • COCO model   – person + motorcycle detected; helmet inferred
                       via HSV/edge heuristic; violation only for riders.
      • Custom model – model has explicit helmet/no_helmet classes;
                       heuristic is skipped entirely.
    """

    def __init__(self):
        self.model       = None
        self.device      = DEVICE
        self.custom_mode = False
        self.class_names = {}
        self._fps_t      = time.time()
        self._fps        = 0.0
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────

    def _load_model(self):
        try:
            from ultralytics import YOLO
            if not os.path.exists(MODEL_PATH):
                logger.info("Downloading yolov10n.pt …")
                self.model = YOLO("yolov10n.pt")
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                self.model.save(MODEL_PATH)
            else:
                self.model = YOLO(MODEL_PATH)

            self.model.to(self.device)

            names = self.model.names
            if isinstance(names, dict):
                self.class_names = {int(k): str(v) for k, v in names.items()}
            elif isinstance(names, list):
                self.class_names = {i: str(n) for i, n in enumerate(names)}

            custom_labels = {"helmet", "no_helmet", "no helmet",
                             "with_helmet", "without_helmet"}
            self.custom_mode = bool(
                custom_labels & {n.lower() for n in self.class_names.values()}
            )
            logger.info("YOLOv10 ready | device=%s | custom_mode=%s | classes=%s",
                        self.device, self.custom_mode,
                        list(self.class_names.values())[:10])
        except Exception as exc:
            logger.error("Model load failed: %s", exc, exc_info=True)
            self.model = None

    # ── Main detect ───────────────────────────────────────────────────

    def detect(self, frame):
        """
        Run inference on one BGR frame.

        Returns
        -------
        detections : list[dict]  label, confidence, box(x1,y1,x2,y2)
        annotated  : np.ndarray  frame with boxes drawn
        violation  : bool        True only when rider + no helmet
        """
        if self.model is None:
            return [], frame.copy(), False

        now = time.time()
        self._fps = 1.0 / max(now - self._fps_t, 1e-6)
        self._fps_t = now

        annotated  = frame.copy()
        detections = []
        violation  = False

        try:
            results = self.model.predict(
                frame,
                conf=CONFIDENCE,
                iou=IOU_THRESHOLD,
                device=self.device,
                verbose=False,
            )[0]

            persons     = []   # (conf, x1,y1,x2,y2)
            motorcycles = []   # (conf, x1,y1,x2,y2)

            # ── Parse raw YOLO boxes ──────────────────────────────────
            print("==== VIOLATION DEBUG ====")
            for box in results.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                name = self.class_names.get(cls, "cls_{}".format(cls))
                print("  Detected: {} ({:.2f}) @ [{},{},{},{}]".format(
                    name, conf, x1, y1, x2, y2))

                if self.custom_mode:
                    nl = name.lower().replace(" ", "_")
                    if nl in ("helmet", "with_helmet"):
                        self._draw(annotated, x1, y1, x2, y2,
                                   "HELMET {:.0%}".format(conf), COLOR_HELMET)
                        detections.append({"label": "helmet", "confidence": conf,
                                           "box": (x1, y1, x2, y2)})
                    elif nl in ("no_helmet", "without_helmet", "no helmet"):
                        self._draw(annotated, x1, y1, x2, y2,
                                   "NO HELMET {:.0%}".format(conf),
                                   COLOR_NO_HELMET, thickness=3)
                        detections.append({"label": "no_helmet", "confidence": conf,
                                           "box": (x1, y1, x2, y2)})
                        violation = True
                    elif nl == "motorcycle":
                        motorcycles.append((conf, x1, y1, x2, y2))
                    elif nl == "person":
                        persons.append((conf, x1, y1, x2, y2))
                else:
                    if cls == PERSON_CLASS:
                        persons.append((conf, x1, y1, x2, y2))
                    elif cls == MOTORCYCLE_CLASS:
                        motorcycles.append((conf, x1, y1, x2, y2))

            person_detected     = len(persons) > 0
            motorcycle_detected = len(motorcycles) > 0
            helmet_detected     = False   # updated below

            print("Person detected    : {}".format(person_detected))
            print("Motorcycle detected: {}".format(motorcycle_detected))

            # ── COCO mode: draw motorcycles ───────────────────────────
            if not self.custom_mode:
                for conf, x1, y1, x2, y2 in motorcycles:
                    self._draw(annotated, x1, y1, x2, y2,
                               "Motorcycle {:.0%}".format(conf), COLOR_MOTORCYCLE)
                    detections.append({"label": "motorcycle", "confidence": conf,
                                       "box": (x1, y1, x2, y2)})

                # ── Per-person: check rider status + helmet ───────────
                for conf, px1, py1, px2, py2 in persons:
                    is_rider = self._is_rider(px1, py1, px2, py2, motorcycles)
                    has_helmet, hdbg = self._check_helmet(
                        frame, px1, py1, px2, py2)

                    if has_helmet:
                        helmet_detected = True

                    print("  Person ({},{}) is_rider={} has_helmet={} | {}".format(
                        px1, py1, is_rider, has_helmet, hdbg))

                    if is_rider:
                        if has_helmet:
                            # Helmeted rider — green box
                            self._draw(annotated, px1, py1, px2, py2,
                                       "HELMET {:.0%}".format(conf),
                                       COLOR_HELMET)
                            detections.append({"label": "helmet",
                                               "confidence": conf,
                                               "box": (px1, py1, px2, py2)})
                        else:
                            # Rider WITHOUT helmet — RED box + violation
                            self._draw(annotated, px1, py1, px2, py2,
                                       "NO HELMET - VIOLATION {:.0%}".format(conf),
                                       COLOR_NO_HELMET, thickness=3)
                            detections.append({"label": "no_helmet",
                                               "confidence": conf,
                                               "box": (px1, py1, px2, py2)})
                            violation = True
                    else:
                        # Pedestrian — orange box, no violation
                        self._draw(annotated, px1, py1, px2, py2,
                                   "Person {:.0%}".format(conf),
                                   COLOR_PEDESTRIAN)
                        detections.append({"label": "person",
                                           "confidence": conf,
                                           "box": (px1, py1, px2, py2)})

            print("Helmet detected    : {}".format(helmet_detected))
            print("Violation          : {}".format(violation))

            # ── Violation banner ──────────────────────────────────────
            if violation:
                cv2.rectangle(annotated, (0, 0),
                              (annotated.shape[1], 64), (0, 0, 180), -1)
                cv2.putText(annotated,
                            "!! VIOLATION – NO HELMET DETECTED !!",
                            (10, 44), cv2.FONT_HERSHEY_DUPLEX,
                            1.0, (255, 255, 255), 2, cv2.LINE_AA)

            # ── FPS overlay ───────────────────────────────────────────
            cv2.putText(annotated,
                        "FPS:{:.0f}  {}".format(self._fps, DEVICE.upper()),
                        (10, annotated.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (180, 180, 180), 1, cv2.LINE_AA)

        except Exception as exc:
            logger.error("Detection error: %s", exc, exc_info=True)
            print("[ERROR] {}".format(exc))

        return detections, annotated, violation

    # ── Rider–motorcycle association ──────────────────────────────────

    @staticmethod
    def _is_rider(px1, py1, px2, py2, motorcycles):
        """
        Return True when the person box overlaps a motorcycle box
        by at least RIDER_OVERLAP_FRAC of the person's height.

        Logic: a rider sits ON the motorcycle, so the lower portion
        of the person box overlaps the upper portion of the bike box.
        We check vertical overlap as a fraction of person height.
        """
        if not motorcycles:
            return False

        person_h = max(1, py2 - py1)

        for _, mx1, my1, mx2, my2 in motorcycles:
            # Horizontal overlap
            h_overlap = max(0, min(px2, mx2) - max(px1, mx1))
            if h_overlap == 0:
                continue

            # Vertical overlap
            v_overlap = max(0, min(py2, my2) - max(py1, my1))
            overlap_frac = v_overlap / person_h

            if overlap_frac >= RIDER_OVERLAP_FRAC:
                return True

            # Also accept: person box is directly above motorcycle
            # (rider's feet touch the top of the bike box)
            gap = my1 - py2   # positive = person above bike
            if 0 <= gap <= person_h * 0.5 and h_overlap > (px2 - px1) * 0.3:
                return True

        return False

    # ── Helmet heuristic ──────────────────────────────────────────────

    def _check_helmet(self, frame, x1, y1, x2, y2):
        """
        Examine the top HEAD_FRAC of the person bounding box.

        Returns (bool has_helmet, str debug_info)
        """
        try:
            bh = y2 - y1
            bw = x2 - x1
            if bh < 20 or bw < 10:
                return False, "box too small"

            head_y2   = y1 + max(1, int(bh * HEAD_FRAC))
            head_crop = frame[max(0, y1):head_y2,
                              max(0, x1):min(frame.shape[1], x2)]

            if head_crop.size == 0 or head_crop.shape[0] < 4:
                return False, "empty crop"

            hsv      = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
            m_dark   = cv2.inRange(hsv, _DARK_LO, _DARK_HI)
            m_color  = cv2.inRange(hsv, _COLOR_LO, _COLOR_HI)
            combined = cv2.bitwise_or(m_dark, m_color)

            total        = float(combined.size)
            pixel_ratio  = np.count_nonzero(combined) / total

            gray         = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
            edges        = cv2.Canny(gray, 40, 120)
            edge_density = np.count_nonzero(edges) / total

            has_helmet = (
                pixel_ratio  >= HELMET_PIXEL_RATIO
                or (edge_density >= EDGE_DENSITY_MIN and pixel_ratio >= 0.08)
            )
            dbg = "px={:.3f} edge={:.3f} helmet={}".format(
                pixel_ratio, edge_density, has_helmet)
            return has_helmet, dbg

        except Exception as exc:
            return False, "err:{}".format(exc)

    # ── Drawing helper ────────────────────────────────────────────────

    @staticmethod
    def _draw(frame, x1, y1, x2, y2, label, color, thickness=2):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
        ly = max(0, y1 - th - 8)
        cv2.rectangle(frame, (x1, ly), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (255, 255, 255), 1, cv2.LINE_AA)
