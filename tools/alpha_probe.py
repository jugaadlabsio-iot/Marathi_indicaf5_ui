# -*- coding: utf-8 -*-
"""
Sweep the merge ratio and listen to what it does to च / ज.

    python tools/alpha_probe.py

Hypothesis being tested
-----------------------
Marathi distinguishes DENTAL च/ज  [ts]/[dz]  from PALATAL च/ज  [tʃ]/[dʒ].
Hindi has only the palatal pair, and IndicF5's base is Hindi-dominant, so a
Hindi-leaning model renders every Marathi च as [tʃ]. The fine-tune on your own
speech would carry the distinction, because you make it.

model_voice_merged50.pt is half base. The merge is what fixed catastrophic
forgetting - but the same dilution that restored the base model's vocabulary
may have pulled च/ज back toward Hindi. If so, the error should shrink as α
rises, and the pure fine-tune should get it right while getting other words
wrong.

This renders one sentence per checkpoint so the difference is audible rather
than argued. There is no ASR that reliably scores dental vs palatal in
Marathi - your ear is the instrument here.

Output: out/_alpha_<name>.wav  - play them in order.
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--device", default="auto",
                 help="'cpu' runs this without touching the GPU, so it can go "
                      "in parallel with a story render")
_ap.add_argument("--threads", type=int, default=0,
                 help="cap torch threads. This box is 4c/8t and a GPU render "
                      "still needs CPU between chunks for reference "
                      "preprocessing - saturating it would slow that down")
_ap.add_argument("--nfe", type=int, default=32)
_A = _ap.parse_args()
if _A.threads:
    os.environ["OMP_NUM_THREADS"] = str(_A.threads)
    os.environ["MKL_NUM_THREADS"] = str(_A.threads)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("mtts_app", HERE / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

if _A.threads:
    import torch
    torch.set_num_threads(_A.threads)
for _m, _n in ((app.pacing, "pacing"), (app.numerals, "numerals")):
    if _m is None:
        sys.exit(f"app.py could not import {_n} - refusing to run a comparison "
                 f"that would be silently unpaced, which is exactly how the "
                 f"first attempt at this sweep was wasted.")
DEVICE = app.DEVICE if _A.device == "auto" else _A.device
print(f"device: {DEVICE}   threads: {_A.threads or 'default'}   nfe: {_A.nfe}")

# Two real sentences from your scripts, picked for च/ज density (8 each).
TEXT = ("तिचा आवाज, तिच्या बांगड्यांचा आवाज, तिच्या चप्पलांचा आवाज.\n"
        "रोहनने जोर लावला, पण कडी सरकतच राहिली, "
        "जसं पलीकडचा जोर रोहनच्या जोरापेक्षा जास्त आहे.")

# Ordered least → most of your own voice.
CANDIDATES = [
    ("merged50_current", "model_voice_merged50.pt"),
    ("merged55",         "model_last_slim_merged55.pt"),
    ("merged60",         "model_last_slim_merged60.pt"),
    ("merged65",         "model_last_slim_merged65.pt"),
    ("finetune_pure",    "model_last_slim.pt"),
]

import soundfile as sf
ref = min(app.list_refs(), key=lambda r: sf.info(r).duration)
ref_txt = app.ref_text_for(ref)
d = app.load_dict()
avail = {Path(c).name: c for c in app.list_ckpts()}

pauses = {"chunk": 250, "flow": 60, "sent": 300, "line": 350,
          "spara": 320, "para": 600, "end": 0}

print("Same sentence, same seed, same reference - only the merge ratio moves.\n")
for name, fname in CANDIDATES:
    ckpt = avail.get(fname)
    if not ckpt:
        print(f"  skip {name}: {fname} not in {app.MODELS_DIR}")
        continue
    t0 = time.time()
    try:
        out, dur, took, n, _ = app.synthesize(
            TEXT, ckpt, ref, ref_txt, 1.0, _A.nfe, 400, pauses,
            True, [[k, v] for k, v in d.items()], True, DEVICE,
            out_name=f"_alpha_{name}",
            on_progress=lambda f, m: None, log=lambda m: None,
            cfg=2.0, pace=1.35, seed=7, seed_per_chunk=False,
            max_secs=8.0, lead_ms=250, tail_ms=500,
            chain=True, chain_reanchor=8, apply_warmth=True)
        print(f"  {name:<18} {out.name:<44} {dur:5.1f}s  ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  {name:<18} FAILED: {e}", flush=True)

print("\nListen for the च in चप्पलांचा and the ज in जोर / जास्त.")
print("If the higher-α files get them right, the merge is the cause and the")
print("fix is a targeted re-merge - not a dictionary of 1575 words.")
