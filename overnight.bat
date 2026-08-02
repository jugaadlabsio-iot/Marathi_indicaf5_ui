@echo off
REM Unattended overnight render. Started by a Windows Scheduled Task so it
REM survives closing the terminal, closing Claude, and logging out of the app.
REM
REM No cooldown pause: measured, this card sheds 0 degrees in 8 idle minutes,
REM so waiting is pure loss. Bottom-up (--reverse) so an unheard story lands
REM first and can be judged early. --max-secs 8.0 keeps whole sentences whole
REM (26%% of chunks were comma-fragments at 3.2).
setlocal
set PROJ=C:\marathi_tts
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
"%PROJ%\venv\Scripts\python.exe" -u "%PROJ%\run_queue.py" ^
  --ckpt "C:\marathi_tts_models\model_voice_merged50.pt" ^
  --seed 7 --lock-seed --reverse ^
  --max-secs 8.0 ^
  >> "%PROJ%\out\queue_run.log" 2>> "%PROJ%\out\queue_run.err.log"
