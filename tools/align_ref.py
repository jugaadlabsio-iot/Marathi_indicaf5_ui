# -*- coding: utf-8 -*-
"""
Cut a reference clip that exactly matches a known ground-truth sentence.
Uses Whisper word timestamps only to find the boundaries; the text we pair
with the audio is the human-written script line, not Whisper's guess.
"""
import os
import numpy as np, soundfile as sf
from faster_whisper import WhisperModel

SRC   = r"D:\marathi_tts_project\wav_clean\s1_vihir.wav"
OUT   = r"D:\marathi_tts_project\ref\ref_aligned.wav"
WIN_S, WIN_E = 700.0, 726.0          # search window around the located passage

# ground truth from the script (कथा १ विहिरीतला आवाज)
GT = "रात्री बारा वाजता पाटलांच्या पडवीत ते लॅपटॉपवर अहवाल लिहीत बसले होते. बाहेर वारा सुटला होता."
FIRST_WORD_HINT = "रात"      # start: first word beginning with this
LAST_WORD_HINT  = "होता"     # end:   last word containing this

os.makedirs(os.path.dirname(OUT), exist_ok=True)
x, sr = sf.read(SRC, dtype="float32", always_2d=True)
x = x.mean(1) if x.shape[1] > 1 else x[:, 0]
seg_audio = x[int(WIN_S * sr):int(WIN_E * sr)]
tmp = OUT.replace(".wav", "_window.wav")
sf.write(tmp, seg_audio, sr, subtype="PCM_16")

m = WhisperModel("large-v3", device="cpu", compute_type="int8")
segs, _ = m.transcribe(tmp, language="mr", word_timestamps=True, vad_filter=False)
words = [w for s in segs for w in (s.words or [])]
print(f"{len(words)} words in window {WIN_S}-{WIN_E}s\n")
for w in words:
    print(f"  {w.start:6.2f}-{w.end:6.2f}  {w.word}")

starts = [i for i, w in enumerate(words) if w.word.strip().startswith(FIRST_WORD_HINT)]
ends = [i for i, w in enumerate(words) if LAST_WORD_HINT in w.word]
if not starts or not ends:
    raise SystemExit("\nCould not locate boundary words - widen the window or adjust hints.")

si = starts[0]
ei = max(i for i in ends if i > si)          # last '...होता' after the start
a = max(0.0, words[si].start - 0.08)
b = min(len(seg_audio) / sr, words[ei].end + 0.20)

clip = seg_audio[int(a * sr):int(b * sr)]
clip = clip / (np.abs(clip).max() + 1e-9) * 0.95
sf.write(OUT, clip, sr, subtype="PCM_16")
open(os.path.splitext(OUT)[0] + ".txt", "w", encoding="utf-8").write(GT)
os.remove(tmp)

print(f"\naligned clip : {OUT}")
print(f"span         : {WIN_S+a:.2f}s -> {WIN_S+b:.2f}s   ({b-a:.2f}s long)")
print(f"paired text  : {GT}")
print(f"whisper heard: {' '.join(w.word.strip() for w in words[si:ei+1])}")
