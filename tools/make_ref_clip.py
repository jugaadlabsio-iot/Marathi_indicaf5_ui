# -*- coding: utf-8 -*-
"""Pick a clean ~9 s reference clip: high speech activity, starts and ends in a pause."""
import numpy as np, soundfile as sf, sys, os

SRC = sys.argv[1] if len(sys.argv) > 1 else r"D:\marathi_tts_project\wav_clean\s1_vihir.wav"
OUT = sys.argv[2] if len(sys.argv) > 2 else r"D:\marathi_tts_project\ref\ref_9s.wav"
TARGET = 9.0          # seconds - well under F5-TTS's ~12 s reference limit
os.makedirs(os.path.dirname(OUT), exist_ok=True)

x, sr = sf.read(SRC, dtype="float32", always_2d=True)
x = x.mean(1) if x.shape[1] > 1 else x[:, 0]

# frame energy in dB relative to peak
fl = int(0.02 * sr)                      # 20 ms frames
n = len(x) // fl
fr = x[:n * fl].reshape(n, fl)
rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
db = 20 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)
speech = db > -35                        # frame is voiced-ish

w = int(TARGET / 0.02)                   # frames per window
best, best_score = None, -1
# slide over the middle 80% of the file (skip intro/outro oddities)
lo, hi = int(n * 0.10), int(n * 0.90) - w
for s in range(lo, max(lo + 1, hi), 5):
    win = speech[s:s + w]
    if len(win) < w:
        break
    score = win.mean()
    # reward starting and ending inside a pause -> clean cut, no clipped word
    edge = (not speech[s]) + (not speech[s + w - 1])
    score += 0.15 * edge
    if score > best_score:
        best_score, best = score, s

a, b = best * fl, (best + w) * fl
clip = x[a:b]
clip = clip / (np.abs(clip).max() + 1e-9) * 0.95      # normalize, avoid clipping
sf.write(OUT, clip, sr, subtype="PCM_16")
print(f"source     : {SRC}")
print(f"clip       : {OUT}")
print(f"start      : {a/sr:.2f}s   duration: {len(clip)/sr:.2f}s @ {sr}Hz")
print(f"speech frames in clip: {100*speech[best:best+w].mean():.0f}%  (want 70-95%)")
