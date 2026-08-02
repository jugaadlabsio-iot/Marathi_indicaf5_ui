@echo off
REM Unattended overnight render. Started by a Windows Scheduled Task so it
REM survives closing the terminal, closing Claude, and logging out of the app.
REM
REM Cools the GPU below 70C before each story (8 min cap).
REM The card idles near 78C, so a lower target is unreachable and would
REM just burn the cap before every story. The card throttles hard at 90C
REM and a hot render measured 3x slower than a cold one, so the pause is not
REM lost time.
setlocal
set PROJ=C:\marathi_tts
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
"%PROJ%\venv\Scripts\python.exe" -u "%PROJ%\run_queue.py" ^
  --ckpt "C:\marathi_tts_models\model_voice_merged50.pt" ^
  --seed 7 --lock-seed ^
  --cool-below 70 --cool-max-min 8 ^
  >> "%PROJ%\out\queue_run.log" 2>> "%PROJ%\out\queue_run.err.log"
