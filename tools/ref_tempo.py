# -*- coding: utf-8 -*-
"""
The reference clip sets the tempo of everything. Find the one that reads
like narration instead of like reporting.

    python tools/ref_tempo.py            # build variants + render a sample of each
    python tools/ref_tempo.py --build    # build the clips only, no GPU

Why this exists
---------------
F5-TTS generates a CONTINUATION of the audio it is conditioned on, so every
chunk of every story inherits the reference clip's speaking rate. The pipeline
has warned about this on every run:

    reference: 41.5 moras in 5.37s -> 8.26 moras/sec
    NOTE: that reference is quick for narration (~8.3 vs ~6 moras/sec)

No parameter fixes it. Roominess only buys time inside a chunk, and above
~1.45 the model fills the surplus with invented speech rather than slowing
down (MAX_ROOMINESS). The rate has to come from the reference itself.

ref_9s.wav is already calmer (6.62 moras/sec) but is too LONG to chain from:
anchored chaining needs 11.0 - ref_sec seconds of headroom for the previous
chunk, and a 9s anchor leaves 2s against a ~5.5s mean chunk. So a calm clip
that is also short is worth more than either clip alone - hence stretching.

Time-stretch is not resampling: pitch is preserved, only articulation slows.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import numpy as np
import soundfile as sf
import librosa
import pacing

p = argparse.ArgumentParser()
p.add_argument("--build", action="store_true", help="build the clips, skip rendering")
p.add_argument("--source", default="ref_short.wav")
a = p.parse_args()

REF = HERE / "ref"
src = REF / a.source
src_txt = src.with_suffix(".txt")
text = src_txt.read_text(encoding="utf-8").strip()

y, sr = sf.read(str(src), dtype="float32")
if y.ndim > 1:
    y = y.mean(axis=1)
base_rate = pacing.moras(text) / (len(y) / sr)
print(f"source: {src.name}  {len(y)/sr:.2f}s  {base_rate:.2f} moras/sec\n")

# rate < 1 slows down. Target roughly 6 moras/sec, the rate app.py calls
# narration pace, plus one either side so the choice is yours and not mine.
VARIANTS = []   # the truncated clips from make_calm_ref.py are used instead

built = []
for name, r in VARIANTS:
    out = REF / f"ref_{name}.wav"
    ys = librosa.effects.time_stretch(y, rate=r, n_fft=2048, hop_length=512)
    peak = float(np.max(np.abs(ys))) or 1.0
    ys = (ys / peak * 0.95).astype(np.float32)
    sf.write(str(out), ys, sr)
    out.with_suffix(".txt").write_text(text, encoding="utf-8")
    dur = len(ys) / sr
    room = 11.0 - dur
    print(f"  {out.name:<20}{dur:5.2f}s  {pacing.moras(text)/dur:5.2f} moras/sec"
          f"   chain headroom {room:4.1f}s"
          f"{'  <- too long to chain a 5.5s chunk' if room < 5.5 else ''}")
    built.append(out)

print(f"\n  {'ref_9s.wav (existing)':<20} 9.00s   6.62 moras/sec   chain headroom  2.0s"
      f"  <- too long to chain a 5.5s chunk")
print(f"  {'ref_short.wav (current)':<20}5.60s   7.42 moras/sec   chain headroom  5.4s")

if a.build:
    sys.exit(0)

# --- render the same passage with each, so tempo is the only variable -------
spec = importlib.util.spec_from_file_location("mtts_app", HERE / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

PASSAGE = """आजी म्हणायची — "त्या झाडावर मुंज्या राहतो. त्याला त्रास देऊ नकोस."

"मुंज्या म्हणजे काय, आजी?"

"एक मुलगा. तुझ्याच वयाचा. खूप वर्षांपूर्वी. त्याचं मुंज झालं — पण लग्न व्हायच्या आधीच तो गेला."

आदित्यला लहानपणी हे गोष्टीसारखं वाटायचं. भुताच्या गोष्टी. आजीच्या."""

ckpt = next(c for c in app.list_ckpts() if "voice_merged" in c.lower())
d = app.load_dict()
pauses = {"chunk": 250, "flow": 60, "sent": 300, "line": 350,
          "spara": 320, "para": 600, "end": 0}

todo = [("A_ref_short_9.57", REF / "ref_short.wav"),
        ("B_nat3_8.53", REF / "ref_nat3.wav"),
        ("C_nat1_8.01", REF / "ref_nat1.wav")]

print("\nRendering the same passage with each reference - tempo is the only change.\n")
for name, refwav in todo:
    if not refwav.exists():
        continue
    try:
        out, dur, took, n, _ = app.synthesize(
            PASSAGE, ckpt, str(refwav), app.ref_text_for(str(refwav)),
            1.0, 32, 400, pauses,
            True, [[k, v] for k, v in d.items()], True, app.DEVICE,
            out_name=f"_tempo_{name}",
            on_progress=lambda f, m: None, log=lambda m: None,
            cfg=2.0, pace=1.35, seed=7, seed_per_chunk=False,
            max_secs=8.0, lead_ms=250, tail_ms=600,
            chain=True, chain_reanchor=8, apply_warmth=True)
        print(f"  {name:<20} {out.name:<46} {dur:5.1f}s", flush=True)
    except Exception as e:
        print(f"  {name:<20} FAILED: {e}", flush=True)

print("\nPick on tempo alone. Whichever you choose, I will make chaining work")
print("around its length rather than the other way round.")
