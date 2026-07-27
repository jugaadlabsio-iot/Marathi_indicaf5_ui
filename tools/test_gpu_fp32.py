# -*- coding: utf-8 -*-
"""Force float32 on CUDA and check the NaN is gone (and how fast it is)."""
import os, time, numpy as np, soundfile as sf, torch
from f5_tts.api import F5TTS

MODELS = r"C:\marathi_tts_models"
CKPT = os.path.join(MODELS, "model_last_slim.pt")
VOCAB = os.path.join(MODELS, "vocab.txt")
REF = r"D:\marathi_tts_project\ref\ref_9s.wav"
REF_TXT = open(REF.replace(".wav", ".txt"), encoding="utf-8").read().strip()
GEN = "ही एक नवीन मराठी भयकथा आहे. रात्रीच्या अंधारात, त्या जुन्या वाड्यात, काहीतरी हालचाल जाणवत होती."
OUT = r"D:\marathi_tts_project\out\generated_gpu_fp32.wav"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

print("building model on cuda...", flush=True)
tts = F5TTS(model="F5TTS_Base", ckpt_file=CKPT, vocab_file=VOCAB, device="cuda")

# what dtype did it pick on its own?
def first_dtype(m):
    try:
        return next(m.parameters()).dtype
    except Exception:
        return "n/a"

for name in ("ema_model", "model", "vocoder"):
    obj = getattr(tts, name, None)
    if obj is not None and hasattr(obj, "parameters"):
        print(f"  {name:10s} dtype BEFORE: {first_dtype(obj)}", flush=True)

# THE FIX: force everything to float32
for name in ("ema_model", "model", "vocoder"):
    obj = getattr(tts, name, None)
    if obj is not None and hasattr(obj, "float"):
        setattr(tts, name, obj.float())
for name in ("ema_model", "model", "vocoder"):
    obj = getattr(tts, name, None)
    if obj is not None and hasattr(obj, "parameters"):
        print(f"  {name:10s} dtype AFTER : {first_dtype(obj)}", flush=True)

print("\ngenerating on GPU (fp32)...", flush=True)
t0 = time.time()
wav, sr, _ = tts.infer(ref_file=REF, ref_text=REF_TXT, gen_text=GEN,
                       file_wave=OUT, remove_silence=False)
dt = time.time() - t0

x = np.asarray(wav, dtype=np.float32)
nan = int(np.isnan(x).sum())
peak = float(np.abs(np.nan_to_num(x)).max())
print(f"\ngenerated in {dt:.1f}s | {len(x)/sr:.2f}s audio | NaN samples: {nan} | peak {peak:.4f}")
if nan or peak == 0:
    print("VERDICT: still broken on GPU -> use CPU for production.")
else:
    rt = (len(x) / sr) / dt
    print(f"VERDICT: GPU fp32 WORKS  ({rt:.2f}x realtime)")
    print(f"wrote {OUT}")
