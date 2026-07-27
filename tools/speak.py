# -*- coding: utf-8 -*-
"""
Generate Marathi speech with the fine-tuned IndicF5 checkpoint, and report
whether the result is real audio, digital silence, or inaudibly quiet.

usage:  python speak.py "मराठी मजकूर इथे"  [--ckpt PATH] [--out PATH]
"""
import argparse, inspect, os, sys
import numpy as np
import soundfile as sf

MODELS = r"C:\marathi_tts_models"
p = argparse.ArgumentParser()
p.add_argument("text", nargs="?",
               default="ही एक नवीन मराठी भयकथा आहे. रात्रीच्या अंधारात, "
                       "त्या जुन्या वाड्यात, काहीतरी हालचाल जाणवत होती.")
p.add_argument("--ckpt", default=os.path.join(MODELS, "model_last.pt"))
p.add_argument("--vocab", default=os.path.join(MODELS, "vocab.txt"))
p.add_argument("--ref", default=r"D:\marathi_tts_project\ref\ref_9s.wav")
p.add_argument("--ref-text", default=None,
               help="exact transcript of the reference clip (required for good output)")
p.add_argument("--out", default=r"D:\marathi_tts_project\out\generated.wav")
p.add_argument("--model", default="F5TTS_Base")
p.add_argument("--device", default="cuda")
args = p.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)

ref_text = args.ref_text
if ref_text is None:
    side = os.path.splitext(args.ref)[0] + ".txt"
    if os.path.exists(side):
        ref_text = open(side, encoding="utf-8").read().strip()
    else:
        sys.exit(f"No reference transcript. Provide --ref-text or create {side}")

print(f"checkpoint : {args.ckpt}")
print(f"vocab      : {args.vocab}")
print(f"reference  : {args.ref}")
print(f"ref_text   : {ref_text[:80]}{'...' if len(ref_text) > 80 else ''}")
print(f"gen_text   : {args.text[:80]}{'...' if len(args.text) > 80 else ''}\n")

import torch
print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}"
      + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
device = args.device if torch.cuda.is_available() else "cpu"
if device != args.device:
    print("!! CUDA unavailable, falling back to CPU (slower but fine)")

from f5_tts.api import F5TTS

# constructor arg names drift between releases - adapt to whatever this one wants
sig = inspect.signature(F5TTS.__init__).parameters
kw = {"ckpt_file": args.ckpt, "vocab_file": args.vocab, "device": device}
if "model" in sig:
    kw["model"] = args.model
elif "model_type" in sig:
    kw["model_type"] = args.model
print("F5TTS init args:", {k: (v if k != "ckpt_file" else "...") for k, v in kw.items()})
tts = F5TTS(**{k: v for k, v in kw.items() if k in sig})

isig = inspect.signature(tts.infer).parameters
ikw = {"ref_file": args.ref, "ref_text": ref_text, "gen_text": args.text}
if "file_wave" in isig:
    ikw["file_wave"] = args.out
if "remove_silence" in isig:
    ikw["remove_silence"] = False
result = tts.infer(**{k: v for k, v in ikw.items() if k in isig})

# older versions return (wav, sr, spec) and may not write the file themselves
wav, sr = None, 24000
if isinstance(result, tuple) and len(result) >= 2:
    wav, sr = result[0], result[1]
    if isinstance(wav, np.ndarray) and not os.path.exists(args.out):
        sf.write(args.out, wav, sr)

if not os.path.exists(args.out) and wav is not None:
    sf.write(args.out, np.asarray(wav), sr)

print(f"\nwrote {args.out}")

# ---- diagnosis -------------------------------------------------------------
x, sr = sf.read(args.out, dtype="float32", always_2d=True)
x = x.mean(1) if x.shape[1] > 1 else x[:, 0]
peak = float(np.abs(x).max()) if len(x) else 0.0
rms = float(np.sqrt((x ** 2).mean())) if len(x) else 0.0
dbfs = 20 * np.log10(peak) if peak > 0 else -999
print(f"duration {len(x)/sr:.2f}s @ {sr}Hz | peak {peak:.6g} ({dbfs:.1f} dBFS) | rms {rms:.6g}")

if peak == 0:
    print("VERDICT: digital silence - model/vocoder produced nothing.")
elif dbfs < -50:
    print("VERDICT: inaudibly quiet but non-zero -> writing normalized copy.")
    sf.write(args.out.replace(".wav", "_normalized.wav"), x / peak * 0.95, sr)
    print("         listen to the *_normalized.wav")
else:
    print("VERDICT: normal audio level - play it.")
