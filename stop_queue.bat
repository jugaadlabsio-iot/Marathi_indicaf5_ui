@echo off
REM Ask the queue to stop AFTER the story it is currently rendering.
REM
REM The runner checks for this file between stories only, so the story in
REM flight always finishes and is written out. Nothing is killed mid-render.
setlocal
set STOPFILE=C:\marathi_tts\queue\STOP
echo stop-requested > "%STOPFILE%"
echo.
echo Stop requested.
echo   The current story will finish and be saved, then the runner exits.
echo   Remaining stories stay in the queue.
echo.
echo   Cancel it   : del "%STOPFILE%"
echo   Watch it    : progress.bat watch
echo.
pause
