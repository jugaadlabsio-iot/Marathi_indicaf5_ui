# -*- coding: utf-8 -*-
"""Why is this machine slow? Run it on the machine in question.

Answers four things, in the order they usually matter:

  1. Is the GPU actually being used, or did PyTorch quietly fall back to CPU?
  2. Do MPS operations fall back to CPU one by one? (Apple Silicon only.)
     PYTORCH_ENABLE_MPS_FALLBACK=1 makes unimplemented ops run on the CPU
     silently. It stops crashes and it can also make a fast machine slower
     than a slow one, with nothing in the logs to say so.
  3. Does half precision work here? Forcing fp32 everywhere came from a
     GTX 16-series bug. Apple Silicon and RTX cards do not have it and pay
     roughly 2x for the workaround.
  4. How long does a real chunk actually take, in each precision?

    python tools/bench_device.py
    python tools/bench_device.py --nfe 32 --runs 3
"""
import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")

p = argparse.ArgumentParser()
p.add_argument("--nfe", type=int, default=32)
p.add_argument("--runs", type=int, default=2)
p.add_argument("--text", default="पाटलांनी एक क्षण थांबलं आणि मागे वळून बघितलं.")
a = p.parse_args()

import torch                                              # noqa: E402
import numpy as np                                        # noqa: E402
import soundfile as sf                                    # noqa: E402
import importlib.util                                     # noqa: E402

spec = importlib.util.spec_from_file_location("mtts_app", HERE / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
import pacing                                             # noqa: E402

print("=" * 62)
print("  DEVICE")
print("=" * 62)
print(f"  torch            : {torch.__version__}")
print(f"  platform         : {sys.platform}")
mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
print(f"  cuda available   : {torch.cuda.is_available()}")
print(f"  mps  available   : {mps}")
if torch.cuda.is_available():
    print(f"  gpu              : {torch.cuda.get_device_name(0)}")
print(f"  app picked       : {app.DEVICE}")
if app.DEVICE == "cpu":
    print("\n  >>> RUNNING ON CPU. That alone explains any slowness.")
    print("  >>> On Apple Silicon, reinstall torch: pip install --force-reinstall torch torchaudio")

fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
if app.DEVICE == "mps":
    print(f"  MPS_FALLBACK     : {fallback or 'unset'}"
          f"{'  (unimplemented ops run on CPU, silently)' if fallback == '1' else ''}")

print()
print("=" * 62)
print("  A SINGLE CHUNK, EACH PRECISION")
print("=" * 62)

ckpts = app.list_ckpts()
if not ckpts:
    sys.exit(f"no checkpoint in {app.MODELS_DIR}")
ckpt = next((c for c in ckpts if "voice_merged" in c.lower()),
            next((c for c in ckpts if "slim" in c.lower()), ckpts[0]))
ref = min(app.list_refs(), key=lambda r: sf.info(r).duration)
ref_txt = app.ref_text_for(ref)
_, ref_norm, ref_sec = app.ref_profile(ref, ref_txt.strip())
rate = pacing.speech_rate(ref_norm, ref_sec)
want = pacing.estimate_seconds(a.text, rate, roominess=1.08 * 1.35)
print(f"  checkpoint : {Path(ckpt).name}")
print(f"  text       : {a.text}")
print(f"  target     : {want:.2f}s of speech, NFE {a.nfe}\n")

results = {}
for precision in ("fp32", "half"):
    app._cache.update(key=None, tts=None)          # force a clean load
    try:
        tts = app.get_tts(ckpt, str(app.MODELS_DIR / "vocab.txt"), app.DEVICE,
                          precision=("fp32" if precision == "fp32" else "half"))
    except Exception as e:
        print(f"  {precision:5} : load failed - {e}")
        continue
    times, bad = [], False
    for i in range(a.runs):
        t0 = time.time()
        wav, sr, _ = tts.infer(gen_text=a.text, ref_file=ref,
                               ref_text=ref_txt.strip(), speed=1.0,
                               nfe_step=a.nfe, cfg_strength=2.0,
                               sway_sampling_coef=-1.0, remove_silence=False,
                               seed=7, fix_duration=ref_sec + want)
        times.append(time.time() - t0)
        w = np.asarray(wav, dtype=np.float32)
        if np.isnan(w).any() or float(np.abs(w).max()) < 1e-4:
            bad = True
    best = min(times)
    results[precision] = None if bad else best
    note = "  <<< NaN or silence, unusable on this device" if bad else ""
    print(f"  {precision:5} : {best:6.2f}s  (best of {a.runs}){note}")

print()
print("=" * 62)
print("  VERDICT")
print("=" * 62)
f32, h = results.get("fp32"), results.get("half")
if h and f32:
    print(f"  half is {f32/h:.2f}x faster and produces valid audio here.")
    print("  Leave precision on 'auto' - the app will use it.")
elif f32 and not h:
    print("  half produces NaN on this device (GTX 16-series trait).")
    print("  fp32 is correct here and the app will keep using it.")
if f32:
    audio = want
    print(f"  realtime factor (fp32): {f32/audio:.1f}x slower than realtime")
    print(f"  a 10-minute story is roughly {f32/audio*10/60:.1f} hours of rendering")
if app.DEVICE == "mps" and fallback == "1":
    print()
    print("  If this is slower than a several-year-old NVIDIA card, the likely")
    print("  cause is per-op CPU fallback. To find out which ops, run:")
    print("      PYTORCH_ENABLE_MPS_FALLBACK=0 python tools/bench_device.py")
    print("  It will crash naming the first unimplemented op instead of")
    print("  silently running it on the CPU.")
