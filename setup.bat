@echo off
setlocal enabledelayedexpansion
title Dataset Analyser — Setup & Launch

echo.
echo  =====================================================
echo   AI Data Scientist ^& ML Mentor — Environment Setup
echo  =====================================================
echo.

:: ── Check Python ────────────────────────────────────────────────────────────
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found. Please install Python 3.10+ and try again.
    pause & exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK] Found Python %PYVER%
echo.

:: ── Create venv if it doesn't exist ─────────────────────────────────────────
if not exist "venv\Scripts\activate.bat" (
    echo  [SETUP] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo  [ERROR] Failed to create virtual environment.
        pause & exit /b 1
    )
    echo  [OK] Virtual environment created.
) else (
    echo  [OK] Virtual environment already exists.
)
echo.

:: ── Activate venv ────────────────────────────────────────────────────────────
call venv\Scripts\activate.bat
echo  [OK] Virtual environment activated.
echo.

:: ── Upgrade pip ──────────────────────────────────────────────────────────────
echo  [SETUP] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo  [OK] pip upgraded.
echo.

:: ── Install requirements ─────────────────────────────────────────────────────
echo  [SETUP] Installing dependencies (this may take 2-3 minutes)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Some packages failed to install. Check the error above.
    pause & exit /b 1
)
echo.
echo  [OK] All dependencies installed successfully!
echo.

:: ── Done ─────────────────────────────────────────────────────────────────────
echo  =====================================================
echo   Setup complete! To run the app in future, use:
echo     run_app.bat
echo  =====================================================
echo.
pause
endlocal
