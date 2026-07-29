#!/bin/bash
# Marathi Story Voice - update an existing Mac install.
#
#   cd ~/Marathi_indicaf5_ui && bash mac/update_mac.sh
#
# Pulls the latest code and copies it into ~/marathi_tts.
# It does NOT touch: models/, ref/, out/, queue/, venv/, or your
# pronunciation.json - your voice, your clips and your audio are safe.
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$HERE/app.py" ]; then ROOT="$HERE"; else ROOT="$(cd "$HERE/.." && pwd)"; fi
HOME_DIR="$HOME/marathi_tts"

if [ ! -d "$HOME_DIR/venv" ]; then
  echo "No install found at $HOME_DIR - run 'bash mac/setup_mac.sh' first."
  exit 1
fi

if [ -d "$ROOT/.git" ]; then
  echo "==> pulling latest"
  git -C "$ROOT" pull --ff-only
  echo "==> now at: $(git -C "$ROOT" describe --tags --always)"
fi

echo
echo "==> copying app files into $HOME_DIR"
for f in app.py translit.py numerals.py run_queue.py; do
  [ -f "$ROOT/$f" ] && cp -v "$ROOT/$f" "$HOME_DIR/"
done
cp -v "$HERE/start_ui.command" "$HOME_DIR/" 2>/dev/null || true
chmod +x "$HOME_DIR/start_ui.command" 2>/dev/null || true

# pronunciation.json is yours - only seed it if you have none yet
if [ ! -f "$HOME_DIR/pronunciation.json" ] && [ -f "$ROOT/pronunciation.json" ]; then
  cp -v "$ROOT/pronunciation.json" "$HOME_DIR/"
else
  echo "   keeping your existing pronunciation.json"
fi

# reference transcripts ship in git; the .wav files never do
mkdir -p "$HOME_DIR/ref"
cp -vn "$ROOT"/ref/*.txt "$HOME_DIR/ref/" 2>/dev/null || true

echo
echo "==> checking the new modules import"
cd "$HOME_DIR"          # import from the install, not from the repo checkout
"$HOME_DIR/venv/bin/python" - <<'PY'
import sys
ok = True
for m in ("numerals", "translit"):
    try:
        __import__(m)
        print(f"   {m}: ok")
    except Exception as e:
        ok = False
        print(f"   {m}: FAILED - {e}")
try:
    import cmudict            # noqa: F401
    print("   cmudict: ok")
except Exception:
    print("   cmudict: missing -> run  ~/marathi_tts/venv/bin/pip install cmudict")
sys.exit(0 if ok else 1)
PY

echo
echo "============================================================"
echo " Updated. Restart the app:"
echo "     ~/marathi_tts/start_ui.command"
echo "     open http://127.0.0.1:7860"
echo "============================================================"
