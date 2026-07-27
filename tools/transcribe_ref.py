# -*- coding: utf-8 -*-
"""Transcribe the reference clip in Marathi and save it next to the wav."""
import os, sys
from faster_whisper import WhisperModel

REF = sys.argv[1] if len(sys.argv) > 1 else r"D:\marathi_tts_project\ref\ref_9s.wav"
SIZE = sys.argv[2] if len(sys.argv) > 2 else "large-v3"

# 4 GB card: keep Whisper on CPU so it never competes with F5-TTS for VRAM
model = WhisperModel(SIZE, device="cpu", compute_type="int8")
segments, info = model.transcribe(REF, language="mr", beam_size=5, vad_filter=False)
text = " ".join(s.text.strip() for s in segments).strip()

out = os.path.splitext(REF)[0] + ".txt"
open(out, "w", encoding="utf-8").write(text)
print(f"clip  : {REF}")
print(f"model : {SIZE} (cpu/int8)")
print(f"text  : {text}")
print(f"saved : {out}")
print("\nCheck the text above matches what you hear. Fix the .txt by hand if a word is wrong -")
print("reference-transcript accuracy matters more than almost anything else at inference time.")
