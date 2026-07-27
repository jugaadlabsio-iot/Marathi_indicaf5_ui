@echo off
REM Overnight batch - no browser required.
REM   run_queue.bat              fast draft (NFE 16)
REM   run_queue.bat --nfe 32     final quality
setlocal
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
echo ============================================================
echo   Marathi Story Voice - overnight queue
echo   Leave this window open. Ctrl+C to stop.
echo ============================================================
"D:\marathi_tts_project\venv\Scripts\python.exe" -u "D:\marathi_tts_project\run_queue.py" %*
echo.
echo Finished. Audio is in D:\marathi_tts_project\out
pause
