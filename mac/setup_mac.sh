#!/bin/bash
# Marathi Story Voice - setup for Apple Silicon (Mac mini M4)
#   usage:  bash setup_mac.sh
set -e

HOME_DIR="$HOME/marathi_tts"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==> installing into $HOME_DIR"
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
  echo "!! plain install failed - almost always 'bitsandbytes', which is a"
  echo "!! TRAINING-only dependency with no Apple Silicon wheel. Installing"
  echo "!! F5-TTS without deps and adding only what inference needs."
  pip install f5-tts --no-deps
  pip install \
    transformers accelerate vocos x-transformers torchdiffeq ema-pytorch \
    librosa soundfile soxr numpy scipy gradio hydra-core omegaconf \
    cached-path pypinyin unidecode tqdm safetensors datasets matplotlib pydub
fi

echo
echo "==> cmudict (English -> Devanagari transliteration)"
pip install cmudict

echo
echo "==> copying app files"
cp -v "$HERE/app.py" "$HOME_DIR/"
cp -v "$HERE/translit.py" "$HOME_DIR/"
cp -v "$HERE/start_ui.command" "$HOME_DIR/"
[ -f "$HERE/pronunciation.json" ] && cp -v "$HERE/pronunciation.json" "$HOME_DIR/"
mkdir -p "$HOME_DIR/ref"
cp -v "$HERE"/ref/* "$HOME_DIR/ref/" 2>/dev/null || true
chmod +x "$HOME_DIR/start_ui.command"

echo
echo "==> checking Apple Silicon acceleration"
python - <<'PY'
import torch, platform
print("torch:", torch.__version__, "| arch:", platform.machine())
ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
print("MPS (GPU) available:", bool(ok), "->", "MPS" if ok else "CPU")
PY

cat <<'EOF'

============================================================
 Setup done.

 STILL NEEDED - the model files, into ~/marathi_tts/models/ :
     model_last.pt      (from the Kaggle notebook Output tab)
     vocab.txt          (from the same run, base/vocab.txt)

 IMPORTANT: the browser saves the checkpoint as "model_last.zip".
 That file IS the .pt - a PyTorch checkpoint is a zip internally.
 Do NOT unzip it. Just rename it:

     mv ~/Downloads/model_last.zip ~/marathi_tts/models/model_last.pt

 Then start the app:
     ~/marathi_tts/start_ui.command      (or double-click it in Finder)
     open http://127.0.0.1:7860
============================================================
EOF
