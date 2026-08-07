# -*- coding: utf-8 -*-
"""
Harvest real breaths from your own master recordings.

    python tools/extract_breaths.py            # build the pool
    python tools/extract_breaths.py --report   # just show what it finds

Why
---
trim_silence() strips the model's own head/tail silence and the pipeline
inserts mathematically exact gaps in its place. Digitally perfect silence
between phrases is one of the strongest tells that speech is synthetic - real
narration breathes, and the ear notices the absence long before it can name it.

These are YOUR breaths, in your room, on your mic, at your mic distance. A
synthesised or stock breath would sit wrong against the voice; one cut from the
same session cannot.

Detection
---------
A breath is not silence and not speech. It sits in a band above room tone and
below voice, it is noise-like rather than pitched, and it lasts 120-450ms. So:
find the room-tone floor per recording, take runs in the band between floor and
speech, and keep the ones whose spectral flatness says "noise" rather than
"vowel". Anything ambiguous is discarded - the pool only needs a few dozen good
ones out of 1.26 hours, so precision matters far more than recall.
"""
import argparse
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

SRC = Path("D:/Audio Recordings")
POOL = HERE / "ref" / "_breaths"

p = argparse.ArgumentParser()
p.add_argument("--report", action="store_true")
p.add_argument("--per-file", type=int, default=8, help="keep at most N per recording")
p.add_argument("--scan-minutes", type=float, default=5.0)
p.add_argument("--min-ms", type=float, default=120.0)
p.add_argument("--max-ms", type=float, default=450.0)
p.add_argument("--min-flatness", type=float, default=0.25,
               help="spectral flatness above this reads as noise, not a vowel")
p.add_argument("--max-flatness", type=float, default=0.95,
               help="ABOVE this it is dither, not breath. These masters are "
                    "noise-gated - 29%% of frames sit at digital silence - and "
                    "silence scores a perfect 1.00, so without this ceiling the "
                    "pool fills with pure silence sorted to the top")
p.add_argument("--lo-db", type=float, default=-62.0,
               help="absolute level window a breath must fall in")
p.add_argument("--hi-db", type=float, default=-38.0)
a = p.parse_args()


def harvest(path):
    info = sf.info(str(path))
    frames = int(a.scan_minutes * 60 * info.samplerate)
    x, sr = sf.read(str(path), frames=frames, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 24000:
        x = librosa.resample(x, orig_sr=sr, target_sr=24000)
        sr = 24000

    win = int(sr * 0.01)
    n = x.size // win
    rms = np.sqrt((x[:n * win].reshape(n, win) ** 2).mean(1))
    db = 20 * np.log10(np.maximum(rms, 1e-9))

    # An absolute window, not one relative to the noise floor. These files are
    # gated, so the 5th percentile is digital silence at about -148dB and
    # anything referenced to it admits the silence itself.
    band = (db > a.lo_db) & (db < a.hi_db)
    out, run = [], None
    for i, b in enumerate(band):
        if b and run is None:
            run = i
        elif not b and run is not None:
            ms = (i - run) * 10.0
            if a.min_ms <= ms <= a.max_ms:
                s0, s1 = run * win, i * win
                seg = x[s0:s1]
                # noise-like, not a held vowel
                flat = float(np.mean(librosa.feature.spectral_flatness(y=seg)))
                if a.min_flatness <= flat <= a.max_flatness:
                    out.append(dict(start=s0 / sr, ms=ms, flat=flat,
                                    peak_db=float(np.max(db[run:i])), seg=seg))
            run = None
    # strongest first, but prefer mid-length ones - very short reads as a click
    # Mid-length first, then noise-likeness. Sorting by flatness alone put
    # dither at the top of every list.
    out.sort(key=lambda c: (abs(c["ms"] - 250), -c["flat"]))
    return out


POOL.mkdir(parents=True, exist_ok=True)
total = 0
print(f"{'recording':<40}{'found':>7}{'kept':>6}")
for w in sorted(SRC.glob("*.wav")):
    try:
        cands = harvest(w)
    except Exception as e:
        print(f"{w.stem[:38]:<40}  failed: {e}")
        continue
    keep = cands[:a.per_file]
    print(f"{w.stem[:38]:<40}{len(cands):>7}{len(keep):>6}")
    if a.report:
        for c in keep[:3]:
            print(f"      {c['start']:7.2f}s  {c['ms']:5.0f}ms  "
                  f"flat {c['flat']:.2f}  peak {c['peak_db']:.0f}dB")
        continue
    for k, c in enumerate(keep):
        seg = c["seg"]
        # short fades so an inserted breath can never click at its own edges
        f = min(int(0.008 * 24000), seg.size // 4)
        if f > 1:
            seg = seg.copy()
            seg[:f] *= np.linspace(0, 1, f)
            seg[-f:] *= np.linspace(1, 0, f)
        sf.write(str(POOL / f"breath_{w.stem[:14].replace(' ', '_')}_{k:02d}.wav"),
                 seg.astype(np.float32), 24000, subtype="PCM_16")
        total += 1

if not a.report:
    print(f"\nwrote {total} breaths to {POOL}")
    print("These are inserted sparsely - only at substantial paragraph breaks,")
    print("roughly one in three, never twice running. See app.py breath_for().")
