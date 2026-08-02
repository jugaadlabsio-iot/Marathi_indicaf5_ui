@echo off
REM Marathi Story Voice - local web UI  (runs from the internal C: drive)
setlocal
set PROJ=C:\marathi_tts
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
echo ============================================================
echo   Marathi Story Voice
echo   Open http://127.0.0.1:7860
echo   Leave this window open. Ctrl+C to stop.
echo ============================================================
"%PROJ%\venv\Scripts\python.exe" -u "%PROJ%\app.py"
echo.
pause
