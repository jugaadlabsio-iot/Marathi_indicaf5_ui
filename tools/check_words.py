# -*- coding: utf-8 -*-
"""Is the last word actually there? Ask an ASR, not your ears.

"Sounds clipped" is a judgement. "The final word is absent from the
transcript" is a fact. This transcribes each rendered wav and reports which
of the expected words survived, so a truncation can be confirmed without a
listening pass - and, more usefully, so a fix can be verified automatically.

    python tools/check_words.py --dir out/tail --text "मी म्हणालो, मी एकटाच आहे."

Whisper is imperfect on short Marathi clips and will misspell; treat a missing
word as a strong hint, and a PRESENT word as reliable - false positives are
far less likely than false negatives here.
"""
import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")

p = argparse.ArgumentParser()
p.add_argument("--dir", default=str(HERE / "out" / "tail"))
p.add_argument("--text", required=True, help="what the clip should say")
p.add_argument("--model", default="small",
               help="whisper size: small is fast, large-v3 is accurate")
a = p.parse_args()

from faster_whisper import WhisperModel                    # noqa: E402

want = [w for w in re.findall(r"[ऀ-ॿ]+", a.text)]
print(f"expected words: {' · '.join(want)}\n")

print("loading whisper...", flush=True)
model = WhisperModel(a.model, device="cpu", compute_type="int8")

files = sorted(Path(a.dir).glob("*.wav"))
if not files:
    sys.exit(f"no wavs in {a.dir}")

for f in files:
    segs, _ = model.transcribe(str(f), language="mr", beam_size=5)
    heard = " ".join(s.text for s in segs).strip()
    heard_words = re.findall(r"[ऀ-ॿ]+", heard)
    # last expected word is the one that matters for a truncation
    last = want[-1] if want else ""
    got_last = any(last in h or h in last for h in heard_words) if last else False
    missing = [w for w in want
               if not any(w in h or h in w for h in heard_words)]
    flag = "OK " if got_last else "!! "
    print(f"{flag}{f.name}")
    print(f"     heard   : {heard}")
    print(f"     last word '{last}': {'present' if got_last else 'MISSING'}")
    if missing:
        print(f"     missing : {' · '.join(missing)}")
    print()

print("'!!' = the final expected word did not survive - that is the truncation.")
