# -*- coding: utf-8 -*-
"""Cut the reference at the sentence pause -> shorter clip = faster generation."""
import numpy as np, soundfile as sf
from pathlib import Path

SRC = Path(r"D:\marathi_tts_project\ref\ref_9s.wav")
DST = Path(r"D:\marathi_tts_project\ref\ref_short.wav")
END = 5.60                      # just inside the 0.78 s pause at 5.42-6.20 s
TEXT = "रात्री बारा वाजता पाटलांच्या पडवीत ते लॅपटॉपवर अहवाल लिहीत बसले होते."

x, sr = sf.read(SRC, dtype="float32", always_2d=True)
x = x.mean(1) if x.shape[1] > 1 else x[:, 0]
clip = x[: int(END * sr)]
clip = clip / (np.abs(clip).max() + 1e-9) * 0.95
sf.write(DST, clip, sr, subtype="PCM_16")
DST.with_suffix(".txt").write_text(TEXT, encoding="utf-8")
print(f"{DST}  {len(clip)/sr:.2f}s")
print(f"text: {TEXT}")
