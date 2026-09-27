# AI-Based Smart Helmet Detection & Automatic Traffic Challan System

> Powered by **YOLOv10 · EasyOCR · Flask · OpenCV · SQLite**

---

## Quick Start (Windows)

```
Double-click  RUN.bat
```

That's it. The script will:
1. Verify Python 3.9+
2. Install all dependencies
3. Download the YOLOv10n model
4. Start the Flask server
5. Open `http://127.0.0.1:5000` in your browser

---

## Manual Setup

```bash
# 1. Install dependencies
python install.py

# 2. Start the server
python app.py

# 3. Open browser
http://127.0.0.1:5000
```

---

## Features

| Feature | Details |
|---|---|
| Live Webcam | Real-time helmet detection via webcam |
| RTSP/IP Camera | Connect any CCTV stream |
| Video Upload | Process MP4/AVI/MOV files |
| Image Upload | Detect from single images |
| OCR Plate Reading | EasyOCR extracts number plate text |
| Auto Challan | Violations saved to SQLite DB |
| Dashboard | Stats, history, evidence images |

---

## Project Structure

```
HelmetDetectionSystem/
├── app.py                  # Flask application
├── install.py              # Auto-installer
├── requirements.txt
├── RUN.bat                 # One-click launcher
├── detection/
│   ├── detect.py           # Main pipeline
│   ├── helmet_detector.py  # YOLOv10 wrapper
│   ├── plate_detector.py   # Contour-based plate finder
│   ├── ocr_reader.py       # EasyOCR wrapper
│   └── violation_manager.py
├── utils/
│   ├── config.py           # Central config
│   ├── database.py         # SQLite helpers
│   └── helpers.py          # Shared utilities
├── templates/              # Jinja2 HTML templates
├── static/                 # CSS / JS
├── models/                 # YOLOv10 weights
├── uploads/                # User uploads
├── outputs/                # Annotated outputs & plates
├── database/               # challan.db
└── logs/                   # app.log
```

---

## Configuration

Edit `utils/config.py` to change:

| Setting | Default | Description |
|---|---|---|
| `CONFIDENCE` | 0.40 | Detection threshold |
| `WEBCAM_INDEX` | 0 | Default webcam |
| `DEVICE` | auto | cuda / cpu |
| `MODEL_PATH` | models/yolov10n.pt | Model weights |

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/live` | Live detection page |
| GET | `/video_feed?src=0` | MJPEG stream |
| POST | `/api/upload-image` | Detect from image |
| POST | `/api/upload-video` | Process video |
| GET | `/api/stats` | JSON statistics |
| GET | `/api/violations` | JSON violation list |

---

## Requirements

- Python 3.9+
- Windows 10/11
- Webcam (optional)
- NVIDIA GPU (optional, auto-detected)
