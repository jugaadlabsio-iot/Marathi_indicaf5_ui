#!/bin/bash
# Marathi Story Voice - setup for Apple Silicon (Mac mini M4)
#
#   from a git clone :  bash mac/setup_mac.sh
#   from the zip     :  bash setup_mac.sh
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
# works whether this script sits beside app.py (zip bundle) or in mac/ (git clone)
if [ -f "$HERE/app.py" ]; then ROOT="$HERE"; else ROOT="$(cd "$HERE/.." && pwd)"; fi
if [ ! -f "$ROOT/app.py" ]; then
  echo "Cannot find app.py near $HERE - run this from inside the cloned repo."
  exit 1
fi
HOME_DIR="$HOME/marathi_tts"

echo "==> source : $ROOT"
echo "==> install: $HOME_DIR"
mkdir -p "$HOME_DIR"/{models,ref,out}

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found.  Install it first:   brew install python@3.11"
  exit 1
fi
echo "==> python: $(python3 --version)"

cd "$HOME_DIR"
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo
echo "==> PyTorch (Apple Silicon / MPS build)"
pip install torch torchaudio

echo
echo "==> F5-TTS"
if ! pip install f5-tts; then
  echo
  echo "!! plain install failed - almost always 'bitsandbytes', a TRAINING-only"
  echo "!! dependency with no Apple Silicon wheel. Installing F5-TTS without"
  echo "!! deps and adding only what inference needs."
  pip install f5-tts --no-deps
  pip install \
    transformers accelerate vocos x-transformers torchdiffeq ema-pytorch \
    librosa soundfile soxr numpy scipy gradio hydra-core omegaconf \
    cached-path pypinyin unidecode tqdm safetensors datasets matplotlib pydub
fi

echo
echo "==> cmudict (English -> Devanagari) + faster-whisper (optional)"
pip install cmudict
pip install faster-whisper || echo "   faster-whisper skipped (auto-transcribe unavailable)"

echo
echo "==> copying app files"
for f in app.py translit.py numerals.py pacing.py run_queue.py review.py pronunciation.json; do
  [ -f "$ROOT/$f" ] && cp -v "$ROOT/$f" "$HOME_DIR/"
done
cp -v "$HERE/start_ui.command" "$HOME_DIR/" 2>/dev/null || true
# operational scripts: progress + unattended runner (Windows has .bat/.ps1)
cp -v "$HERE/progress.sh" "$HERE/overnight.sh" "$HOME_DIR/" 2>/dev/null || true
chmod +x "$HOME_DIR/progress.sh" "$HOME_DIR/overnight.sh" 2>/dev/null || true
chmod +x "$HOME_DIR/start_ui.command" 2>/dev/null || true
mkdir -p "$HOME_DIR/ref"
cp -v "$ROOT"/ref/* "$HOME_DIR/ref/" 2>/dev/null || true

# the tools directory: merge_ckpt.py, repair.py, probe.py and the rest
mkdir -p "$HOME_DIR/tools"
cp -v "$ROOT"/tools/*.py "$HOME_DIR/tools/" 2>/dev/null || true

echo
echo "==> checking Apple Silicon acceleration"
python - <<'PY'
import torch, platform
print("torch:", torch.__version__, "| arch:", platform.machine())
ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
print("MPS (GPU) available:", bool(ok), "->", "MPS" if ok else "CPU")
PY

# --- report what is still missing ------------------------------------------
MODELS_OK=0; REF_OK=0
ls "$HOME_DIR"/models/*.pt >/dev/null 2>&1 && MODELS_OK=1
ls "$HOME_DIR"/ref/*.wav   >/dev/null 2>&1 && REF_OK=1

echo
echo "============================================================"
echo " Setup complete."

if [ "$MODELS_OK" -eq 0 ]; then
cat <<'EOF'

 STILL NEEDED - model files, into ~/marathi_tts/models/ :
     model_last.pt     (Kaggle Output: ckpts/marathi_voice/model_last.pt)
     vocab.txt         (Kaggle Output: base/vocab.txt)

 IMPORTANT: the browser saves the checkpoint as "model_last.zip".
 That file already IS the .pt - a PyTorch checkpoint is a zip internally.
 Do NOT unzip it, just rename it:

     mv ~/Downloads/model_last.zip ~/marathi_tts/models/model_last.pt
EOF
fi

if [ "$REF_OK" -eq 0 ]; then
cat <<'EOF'

 STILL NEEDED - a reference clip, into ~/marathi_tts/ref/ :
   Voice audio is deliberately NOT in the git repo. Either

   (a) copy ref_short.wav AND ref_short.txt across from the PC, or
   (b) record one in the app: "Reference clips" tab -> record 6-10 s of
       calm narration, type the exact transcript, Save.

   Always keep each .wav paired with a .txt of the exact words spoken.
EOF
fi

cat <<'EOF'

 Start it:
     ~/marathi_tts/start_ui.command      (or double-click in Finder)
     open http://127.0.0.1:7860
============================================================
EOF
