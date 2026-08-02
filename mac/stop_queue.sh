#!/bin/bash
# Ask the queue to stop AFTER the story it is currently rendering.
# The runner checks between stories only, so nothing is killed mid-render.
STOPFILE="${MTTS_HOME:-$HOME/marathi_tts}/queue/STOP"
mkdir -p "$(dirname "$STOPFILE")"
echo "stop-requested" > "$STOPFILE"
echo "Stop requested."
echo "  The current story will finish and be saved, then the runner exits."
echo "  Remaining stories stay in the queue."
echo
echo "  Cancel it : rm '$STOPFILE'"
echo "  Watch it  : bash progress.sh watch"
