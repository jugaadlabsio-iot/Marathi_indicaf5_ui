# -*- coding: utf-8 -*-
import numpy as np, soundfile as sf

p = r"D:\marathi_tts_project\out\generated.wav"
x, sr = sf.read(p, dtype="float32", always_2d=True)
x = x[:, 0]
print("first 20 samples:", np.round(x[:20], 4))
u, c = np.unique(np.round(x, 4), return_counts=True)
print("distinct values:", len(u))
print("top values:", [(float(v), int(n)) for v, n in
                      sorted(zip(u, c), key=lambda t: -t[1])[:6]])
print("min %.4f  max %.4f  mean %.4f" % (x.min(), x.max(), x.mean()))
d = np.diff(x)
print("sign flips between consecutive samples: %.1f%%" % (100 * (d != 0).mean()))
# also read raw int16 to see if libsndfile clamped something
raw, _ = sf.read(p, dtype="int16", always_2d=True)
print("int16 min/max:", raw.min(), raw.max())
