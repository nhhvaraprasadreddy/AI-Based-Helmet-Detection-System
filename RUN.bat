@echo off
title AI Helmet Detection System
color 0A
cls

REM Switch to the folder that contains this .bat file.
REM This fixes the file-not-found error on OneDrive / Desktop shortcuts.
cd /d "%~dp0"

echo.
echo  =====================================================
echo   AI-Based Helmet Detection ^& Challan System
echo   Powered by YOLOv10 + EasyOCR + Flask
echo  =====================================================
echo  Working directory: %CD%
echo.

REM 1. Verify Python is installed and on PATH
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo  [ERROR] Python not found in PATH.
    echo  Install Python 3.9+ from https://www.python.org/downloads/
    echo  Tick Add Python to PATH during installation.
    echo.
    pause
    exit /b 1
)
echo  [OK] Python detected:
python --version
echo.

REM 2. Check install.py exists before calling it
IF NOT EXIST "%~dp0install.py" (
    echo  [ERROR] install.py not found in %~dp0
    echo  Re-download the project and keep all files together.
    echo.
    pause
    exit /b 1
)

REM 3. Run installer
echo  [1/3] Running installer...
python "%~dp0install.py"
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo  [ERROR] Installation failed. Run manually: python install.py
    echo.
    pause
    exit /b 1
)
echo  [OK] Installation complete.
echo.

REM 4. Check app.py exists before launching Flask
IF NOT EXIST "%~dp0app.py" (
    echo  [ERROR] app.py not found in %~dp0
    echo  Re-download the project.
    echo.
    pause
    exit /b 1
)

REM 5. Open browser after 4-second delay
start /b cmd /c "timeout /t 4 >nul && start http://127.0.0.1:5000"

REM 6. Start Flask server
echo  [2/3] Starting Flask server...
echo  [3/3] Open browser at: http://127.0.0.1:5000
echo  Press Ctrl+C to stop.
echo.
python "%~dp0app.py"

echo.
echo  Server stopped.
pause