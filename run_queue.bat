@echo off
REM Overnight batch - no browser required.
REM   run_queue.bat              fast draft (NFE 16)
REM   run_queue.bat --nfe 32     final quality
setlocal
set PROJ=C:\marathi_tts
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
echo ============================================================
echo   Marathi Story Voice - overnight queue
echo   Leave this window open. Ctrl+C to stop.
echo ============================================================
"%PROJ%\venv\Scripts\python.exe" -u "%PROJ%\run_queue.py" %*
echo.
echo Finished. Audio is in %PROJ%\out
pause
