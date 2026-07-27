# -*- coding: utf-8 -*-
"""Strip optimizer state from a training checkpoint -> small, fast-loading inference file."""
import os, sys, torch

SRC = sys.argv[1] if len(sys.argv) > 1 else r"C:\marathi_tts_models\model_last.pt"
DST = sys.argv[2] if len(sys.argv) > 2 else r"C:\marathi_tts_models\model_last_slim.pt"

if os.path.exists(DST):
    print(f"already exists: {DST} ({os.path.getsize(DST)/1e9:.2f} GB)")
    raise SystemExit(0)

print(f"loading {SRC} ({os.path.getsize(SRC)/1e9:.2f} GB) - this takes a minute...", flush=True)
ck = torch.load(SRC, map_location="cpu", weights_only=True)
print("top-level keys:", list(ck.keys()), flush=True)

out = {}
if "ema_model_state_dict" in ck:
    out["ema_model_state_dict"] = ck["ema_model_state_dict"]
if "model_state_dict" in ck:
    out["model_state_dict"] = ck["model_state_dict"]
if not out:
    raise SystemExit("no model weights found in checkpoint")

torch.save(out, DST)
print(f"saved {DST} ({os.path.getsize(DST)/1e9:.2f} GB)", flush=True)
print("kept:", list(out.keys()), flush=True)
