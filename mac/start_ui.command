#!/bin/bash
# Double-click this in Finder to start the UI on the Mac.
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MTTS_HOME="$HOME/marathi_tts"
source "$HOME/marathi_tts/venv/bin/activate"
echo "============================================================"
echo "  Marathi Story Voice  ->  http://127.0.0.1:7860"
echo "  (leave this window open; Ctrl+C to stop)"
echo "============================================================"
python "$HOME/marathi_tts/app.py"
