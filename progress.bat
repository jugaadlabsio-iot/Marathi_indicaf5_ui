@echo off
REM Double-click to see how the overnight queue is doing.
REM   progress.bat          one snapshot
REM   progress.bat watch    live, refreshing every 30s
if /I "%1"=="watch" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0progress.ps1" -Watch
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0progress.ps1"
  echo.
  pause
)
