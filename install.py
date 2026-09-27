# ============================================================
# install.py - Auto-installer for Helmet Detection System
# ============================================================
# IMPORTANT: This file must be 100 % self-contained.
# It must NOT import any project module (utils, detection, etc.)
# because those modules depend on packages that may not yet exist.
# ============================================================

import sys
import os
import subprocess
import platform
import shutil

# ── Always work relative to this file's own directory ────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Directories to create ─────────────────────────────────────────────
DIRS = [
    "models",
    "dataset",
    os.path.join("uploads", "videos"),
    os.path.join("uploads", "images"),
    os.path.join("outputs", "violations"),
    os.path.join("outputs", "plates"),
    "database",
    os.path.join("static", "css"),
    os.path.join("static", "js"),
    "templates",
    "detection",
    "utils",
    "logs",
]

# ── Packages to install ───────────────────────────────────────────────
# torch / torchvision are handled separately (CPU vs CUDA index URL)
PACKAGES = [
    "flask",
    "werkzeug",
    "numpy",
    "pandas",
    "pillow",
    "opencv-python",
    "easyocr",
    "ultralytics",
]

REQUIREMENTS_CONTENT = (
    "torch>=2.0.0\n"
    "torchvision>=0.15.0\n"
    "ultralytics>=8.2.0\n"
    "opencv-python>=4.8.0\n"
    "easyocr>=1.7.0\n"
    "flask>=3.0.0\n"
    "numpy>=1.24.0\n"
    "pandas>=2.0.0\n"
    "pillow>=10.0.0\n"
    "werkzeug>=3.0.0\n"
)


# ── Helpers ───────────────────────────────────────────────────────────

def banner(msg):
    line = "=" * 56
    print("\n{}\n  {}\n{}".format(line, msg, line))


def run_pip(*args):
    """Run a pip command via the current Python interpreter."""
    cmd = [sys.executable, "-m", "pip"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr


def is_installed(package_name):
    """Return True if the package can be imported / found by pip."""
    # Normalise: opencv-python -> cv2, pillow -> PIL
    import_map = {
        "opencv-python": "cv2",
        "pillow":        "PIL",
        "ultralytics":   "ultralytics",
        "easyocr":       "easyocr",
        "werkzeug":      "werkzeug",
    }
    import_name = import_map.get(package_name.lower(), package_name.lower())
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def cuda_available():
    """Check CUDA without importing torch (uses nvidia-smi)."""
    return shutil.which("nvidia-smi") is not None


# ── Steps ─────────────────────────────────────────────────────────────

def step_check_python():
    banner("Step 1/5 — Checking Python version")
    v = sys.version_info
    print("  Python {}.{}.{} on {}".format(v.major, v.minor, v.micro, platform.system()))
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        print("  [ERROR] Python 3.9 or newer is required.")
        print("  Download: https://www.python.org/downloads/")
        sys.exit(1)
    print("  [OK] Python version is compatible.")


def step_upgrade_pip():
    banner("Step 2/5 — Upgrading pip")
    ok, _, err = run_pip("install", "--upgrade", "pip", "--quiet")
    if ok:
        print("  [OK] pip upgraded.")
    else:
        print("  [WARN] pip upgrade failed (non-fatal): {}".format(err.strip()))


def step_install_torch():
    banner("Step 3/5 — Installing PyTorch")
    # Check if torch is already installed
    try:
        import torch
        print("  [OK] torch {} already installed.".format(torch.__version__))
        return
    except ImportError:
        pass

    use_cuda = cuda_available()
    if use_cuda:
        print("  NVIDIA GPU detected — installing CUDA build of torch …")
        index_url = "https://download.pytorch.org/whl/cu118"
    else:
        print("  No GPU detected — installing CPU-only torch (smaller download) …")
        index_url = "https://download.pytorch.org/whl/cpu"

    for pkg in ["torch", "torchvision"]:
        print("  Installing {} …".format(pkg), end=" ", flush=True)
        ok, _, err = run_pip("install", pkg,
                             "--index-url", index_url,
                             "--quiet")
        print("[OK]" if ok else "[FAILED] {}".format(err.strip()[:120]))


def step_install_packages():
    banner("Step 4/5 — Installing dependencies")
    run_pip("install", "--upgrade", "pip", "--quiet")   # ensure pip is fresh

    for pkg in PACKAGES:
        already = is_installed(pkg)
        print("  {:30s} {}".format(pkg, "[already installed]" if already else "installing …"),
              end="" if not already else "\n", flush=True)
        if already:
            continue
        ok, _, err = run_pip("install", pkg, "--quiet")
        print(" [OK]" if ok else " [FAILED] {}".format(err.strip()[:120]))


def step_create_dirs():
    banner("Step 5/5 — Creating project directories")
    for d in DIRS:
        full = os.path.join(BASE_DIR, d)
        os.makedirs(full, exist_ok=True)
        print("  [OK] {}".format(d))


def step_write_requirements():
    path = os.path.join(BASE_DIR, "requirements.txt")
    with open(path, "w") as fh:
        fh.write(REQUIREMENTS_CONTENT)
    print("\n  [OK] requirements.txt written.")


def step_download_model():
    banner("Bonus — Checking YOLOv10 model weights")
    model_path = os.path.join(BASE_DIR, "models", "yolov10n.pt")
    if os.path.exists(model_path):
        print("  [OK] yolov10n.pt already present.")
        return

    print("  Downloading yolov10n.pt via ultralytics (first run only) …")
    try:
        from ultralytics import YOLO
        # YOLO() auto-downloads to its cache; we then copy to our models/ dir
        m = YOLO("yolov10n.pt")
        # The file lands in the CWD or ultralytics cache
        candidates = [
            "yolov10n.pt",
            os.path.join(os.path.expanduser("~"), ".config", "Ultralytics", "yolov10n.pt"),
        ]
        for src in candidates:
            if os.path.exists(src):
                shutil.copy2(src, model_path)
                print("  [OK] Model saved to models/yolov10n.pt")
                return
        print("  [OK] Model cached by ultralytics (will load on first detection).")
    except Exception as exc:
        print("  [WARN] Could not pre-download model: {}".format(exc))
        print("  The model will be downloaded automatically on first detection run.")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 56)
    print("  Helmet Detection System — Auto Installer")
    print("  Platform : {} {}".format(platform.system(), platform.release()))
    print("  Base dir : {}".format(BASE_DIR))
    print("=" * 56)

    step_check_python()
    step_upgrade_pip()
    step_install_torch()
    step_install_packages()
    step_create_dirs()
    step_write_requirements()
    step_download_model()

    banner("Installation Complete!")
    print("  Start the system:  python app.py")
    print("  Or double-click:   RUN.bat")
    print("  Then open:         http://127.0.0.1:5000\n")


if __name__ == "__main__":
    main()
