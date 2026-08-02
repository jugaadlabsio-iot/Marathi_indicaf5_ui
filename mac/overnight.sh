#!/bin/bash
# Unattended overnight render. The Mac counterpart of overnight.bat.
#
#   bash mac/overnight.sh              run in the foreground
#   bash mac/overnight.sh detach       survive closing the terminal
#
# Apple Silicon does not thermally collapse the way the Windows box's GTX 1650
# does, so there is no cooldown pause here - it exists on Windows only because
# that card idles near 79C and throttles within minutes.
set -e
PROJ="${MTTS_HOME:-$HOME/marathi_tts}"
cd "$PROJ"

if [ ! -x "$PROJ/venv/bin/python" ]; then
  echo "No install at $PROJ - run mac/setup_mac.sh first."
  exit 1
fi

CKPT="$PROJ/models/model_voice_merged50.pt"
if [ ! -f "$CKPT" ]; then
  echo "Missing $CKPT"
  echo "Copy it from the PC, or rebuild it - see 'Get the model files' in README_MAC.md."
  echo "Falling back to whatever run_queue.py picks."
  CKPT=""
fi

ARGS=(--seed 7 --lock-seed)
[ -n "$CKPT" ] && ARGS+=(--ckpt "$CKPT")

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTORCH_ENABLE_MPS_FALLBACK=1

mkdir -p out
if [ "$1" = "detach" ]; then
  # nohup + disown so it outlives the terminal, the ssh session, and the app
  nohup "$PROJ/venv/bin/python" -u run_queue.py "${ARGS[@]}" \
      >> out/queue_run.log 2>> out/queue_run.err.log &
  disown
  echo "started detached, pid $!"
  echo "  progress : bash mac/progress.sh"
  echo "  log      : $PROJ/out/queue_run.log"
  echo "  stop     : pkill -f run_queue.py"
else
  "$PROJ/venv/bin/python" -u run_queue.py "${ARGS[@]}" \
      2>&1 | tee -a out/queue_run.log
fi
