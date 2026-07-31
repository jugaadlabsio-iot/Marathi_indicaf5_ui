# -*- coding: utf-8 -*-
"""Audit the per-chunk wavs from a finished run.

Two questions, both answerable from what is already on disk:

  1. Did chunks get cut off mid-word? A chunk that is still LOUD at its final
     moment was still talking when its allotted time ran out. That is the
     "end word clipped" complaint, and it is measurable rather than a matter
     of opinion.

  2. How long were the chunks? The probe renders short phrases and gets every
     phoneme right; the stories render much longer ones and do not. If the
     story chunks are far longer, chunk length is the variable to change.

    python tools/analyze_chunks.py
    python tools/analyze_chunks.py --dir out/parts/04_night_doctor_20260731-030741
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np                                        # noqa: E402
import soundfile as sf                                    # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--dir", default=None, help="one run dir; default = all recent")
p.add_argument("--fade-ms", type=float, default=6.0,
               help="the fade applied at write time, skipped when measuring")
p.add_argument("--window-ms", type=float, default=40.0,
               help="how much audio before the fade counts as 'the ending'")
a = p.parse_args()

parts = HERE / "out" / "parts"
dirs = [Path(a.dir)] if a.dir else sorted(
    [d for d in parts.iterdir() if d.is_dir()],
    key=lambda d: d.stat().st_mtime, reverse=True)[:10]

print(f"{'run':44} {'n':>4} {'cut':>5} {'%':>5} {'median s':>9} {'max s':>7}")
print("-" * 80)

grand_n = grand_cut = 0
worst = []
for d in dirs:
    wavs = sorted(d.glob("*.wav"))
    if not wavs:
        continue
    durs, cut = [], 0
    for w in wavs:
        try:
            x, sr = sf.read(w)
        except Exception:
            continue
        x = np.asarray(x, dtype=np.float32)
        if x.size < sr // 20:
            continue
        durs.append(len(x) / sr)
        nf = int(sr * a.fade_ms / 1000)          # the fade we applied
        nw = int(sr * a.window_ms / 1000)
        tail = x[max(0, len(x) - nf - nw): len(x) - nf]
        if tail.size < 8:
            continue
        peak = float(np.abs(x).max()) or 1e-9
        tail_rms = float(np.sqrt((tail ** 2).mean()))
        # still going at full voice when the clock ran out
        rel_db = 20 * np.log10(max(tail_rms, 1e-9) / peak)
        if rel_db > -18.0:
            cut += 1
            worst.append((rel_db, d.name, w.name))
    if not durs:
        continue
    grand_n += len(durs)
    grand_cut += cut
    print(f"{d.name[:44]:44} {len(durs):4d} {cut:5d} "
          f"{cut/len(durs)*100:4.0f}% {np.median(durs):9.2f} {max(durs):7.2f}")

print("-" * 80)
if grand_n:
    print(f"{'TOTAL':44} {grand_n:4d} {grand_cut:5d} {grand_cut/grand_n*100:4.0f}%")
    print()
    print("'cut' = the last 40ms before the fade is within 18 dB of the chunk's")
    print("peak, i.e. the chunk ended at full voice instead of trailing off.")
    worst.sort(reverse=True)
    if worst:
        print("\nloudest endings (most likely truncated):")
        for db, run, name in worst[:12]:
            print(f"  {db:6.1f} dB below peak   {run}/{name}")
