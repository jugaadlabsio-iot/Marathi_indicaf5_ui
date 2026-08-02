#!/bin/bash
# Queue progress at a glance. The Mac counterpart of progress.ps1 on Windows.
#
#   bash ~/marathi_tts/tools/../mac/progress.sh        one snapshot
#   bash mac/progress.sh watch                          refresh every 30s
#
# Checks whether the runner is ACTUALLY alive rather than trusting the log.
# On Windows a run once died silently while its last log line still read
# "cooling before ..." - it looked healthy and had been dead for minutes.
PROJ="${MTTS_HOME:-$HOME/marathi_tts}"
LOG="$PROJ/out/queue_run.log"

snapshot() {
  echo "============================================================"
  echo "  Marathi Story Voice - queue progress        $(date +%H:%M:%S)"
  echo "============================================================"

  if [ ! -f "$LOG" ]; then
    echo "  no run log at $LOG - the queue runner has not been started"
    echo "============================================================"
    return
  fi

  if pgrep -f "run_queue.py" >/dev/null 2>&1; then
    echo "  status    : RUNNING (pid $(pgrep -f run_queue.py | head -1))"
  elif grep -q "^done:" "$LOG" 2>/dev/null; then
    echo "  status    : FINISHED"
  else
    echo "  status    : NOT RUNNING - it stopped or was closed"
  fi

  local cur ok fail chunk eta
  cur=$(grep -E '^\[[0-9]+/[0-9]+\]' "$LOG" | tail -1)
  ok=$(grep -c '^    OK ' "$LOG" 2>/dev/null || echo 0)
  fail=$(grep -c '^    FAILED:' "$LOG" 2>/dev/null || echo 0)
  [ -n "$cur" ] && echo "  story     : ${cur#\[}"
  echo "  completed : $ok   failed: $fail"

  chunk=$(grep -oE 'Chunk [0-9]+/[0-9]+' "$LOG" | tail -1)
  eta=$(grep -oE '~[0-9.]+ min left' "$LOG" | tail -1)
  [ -n "$chunk" ] && echo "  chunk     : ${chunk#Chunk } ${eta:+ ($eta)}"

  # how fast chunks are actually landing, from the newest run folder
  local d n
  d=$(ls -dt "$PROJ"/out/parts/*/ 2>/dev/null | head -1)
  if [ -n "$d" ]; then
    n=$(ls "$d"*.wav 2>/dev/null | wc -l | tr -d ' ')
    echo "  parts     : $(basename "$d")  ($n chunks on disk)"
    # warn if nothing has been written recently while it claims to be running
    if [ "$n" -gt 0 ]; then
      local age
      age=$(( ($(date +%s) - $(stat -f %m "$d" 2>/dev/null || stat -c %Y "$d")) / 60 ))
      [ "$age" -gt 5 ] && echo "  WARNING   : nothing written for ${age} min - it may be stuck"
    fi
  fi

  echo "  queued    : $(ls "$PROJ"/queue/*.txt 2>/dev/null | wc -l | tr -d ' ') story(ies)"

  echo
  echo "  finished audio (newest first):"
  ls -t "$PROJ"/out/*.wav 2>/dev/null | head -5 | while read -r f; do
    echo "    $(basename "$f")"
  done
  echo "============================================================"
}

if [ "$1" = "watch" ]; then
  echo "Watching every 30s. Ctrl-C to stop."
  while true; do clear; snapshot; sleep 30; done
else
  snapshot
fi
