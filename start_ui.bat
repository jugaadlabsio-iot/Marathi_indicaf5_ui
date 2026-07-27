@echo off
REM Marathi Story Voice - local web UI
setlocal
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
echo ============================================================
echo   Marathi Story Voice
echo   Open http://127.0.0.1:7860
echo   Leave this window open. Ctrl+C to stop.
echo ============================================================
"D:\marathi_tts_project\venv\Scripts\python.exe" -u "D:\marathi_tts_project\app.py"
echo.
pause
