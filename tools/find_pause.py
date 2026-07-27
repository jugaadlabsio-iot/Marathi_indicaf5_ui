# -*- coding: utf-8 -*-
"""List silence gaps in a clip so we can cut on a sentence boundary."""
import sys, numpy as np, soundfile as sf

p = sys.argv[1] if len(sys.argv) > 1 else r"D:\marathi_tts_project\ref\ref_9s.wav"
x, sr = sf.read(p, dtype="float32", always_2d=True)
x = x.mean(1) if x.shape[1] > 1 else x[:, 0]
fl = int(0.02 * sr)
n = len(x) // fl
rms = np.sqrt((x[:n*fl].reshape(n, fl) ** 2).mean(1) + 1e-12)
db = 20 * np.log10(rms / rms.max() + 1e-12)
quiet = db < -32

runs, s = [], None
for i, q in enumerate(quiet):
    if q and s is None:
        s = i
    elif not q and s is not None:
        if (i - s) * 0.02 >= 0.12:
            runs.append((s * 0.02, i * 0.02))
        s = None
if s is not None:
    runs.append((s * 0.02, n * 0.02))

print(f"{p}  ({len(x)/sr:.2f}s)")
print("pauses >=0.12s:")
for a, b in runs:
    print(f"   {a:5.2f}s -> {b:5.2f}s   ({b-a:.2f}s)")
