@echo off
setlocal enabledelayedexpansion
title AI Data Scientist Mentor

:: ── Check venv exists ────────────────────────────────────────────────────────
if not exist "venv\Scripts\activate.bat" (
    echo  [ERROR] Virtual environment not found.
    echo  Please run setup.bat first!
    pause & exit /b 1
)

:: ── Activate venv ────────────────────────────────────────────────────────────
call venv\Scripts\activate.bat

:: ── Launch Streamlit ─────────────────────────────────────────────────────────
echo.
echo  =====================================================
echo   Launching AI Data Scientist ^& ML Mentor...
echo   Open your browser at: http://localhost:8501
echo   Press Ctrl+C to stop the server.
echo  =====================================================
echo.

streamlit run app.py --server.headless false --browser.gatherUsageStats false

endlocal
