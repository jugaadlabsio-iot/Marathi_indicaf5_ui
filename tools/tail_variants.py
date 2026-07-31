# -*- coding: utf-8 -*-
"""Which stage is eating the last word?

A clipped ending can come from four different places, and they need different
fixes, so guessing is a waste of a render:

  1. the duration budget    - the canvas ends before the word does
  2. trim_silence           - we cut a quiet final vowel off ourselves
  3. fade_edges             - the 6ms fade lands on the last phoneme
  4. the model              - it simply stops early no matter how much room

This renders the same phrase with each stage turned off in turn, plus a couple
of longer canvases, and writes one clearly-named wav per variant. Listen and
say which one is complete; that names the culprit in a single pass.

    python tools/tail_variants.py
    python tools/tail_variants.py --text "मी म्हणालो, मी एकटाच आहे."
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")

import importlib.util                                     # noqa: E402
import numpy as np                                        # noqa: E402
import soundfile as sf                                    # noqa: E402

spec = importlib.util.spec_from_file_location("mtts_app", HERE / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
import pacing                                             # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--text", default="मी म्हणालो, मी एकटाच आहे.")
p.add_argument("--ckpt", default=None)
p.add_argument("--nfe", type=int, default=32)
p.add_argument("--pace", type=float, default=1.2)
p.add_argument("--device", default="auto")
a = p.parse_args()

out_dir = HERE / "out" / "tail"
out_dir.mkdir(parents=True, exist_ok=True)
for old in out_dir.glob("*.wav"):
    old.unlink()

ckpt = a.ckpt or next((c for c in app.list_ckpts() if "slim" in c.lower()),
                      app.list_ckpts()[0])
ref = min(app.list_refs(), key=lambda r: sf.info(r).duration)
ref_txt = app.ref_text_for(ref)
_, ref_norm, ref_sec = app.ref_profile(ref, ref_txt.strip())
rate = pacing.speech_rate(ref_norm, ref_sec)
device = app.DEVICE if a.device == "auto" else a.device
tts = app.get_tts(ckpt, str(app.MODELS_DIR / "vocab.txt"), device)

base = pacing.estimate_seconds(a.text, rate, roominess=1.08 * a.pace)
print(f"text     : {a.text}")
print(f"reference: {ref_sec:.2f}s   rate {rate:.2f} moras/sec")
print(f"estimate : {base:.2f}s of speech\n")


def render(fix, seed=7):
    kw = dict(ref_file=ref, ref_text=ref_txt.strip(), speed=1.0,
              nfe_step=a.nfe, cfg_strength=2.0, sway_sampling_coef=-1.0,
              remove_silence=False, seed=seed)
    if fix is not None:
        kw["fix_duration"] = ref_sec + fix
    w, sr, _ = tts.infer(gen_text=a.text, **kw)
    return np.asarray(w, dtype=np.float32), sr


def report(name, w, sr):
    path = out_dir / f"{name}.wav"
    peak = float(np.abs(w).max()) or 1.0
    sf.write(path, w / peak * 0.95, sr)
    n = int(sr * 0.04)
    head = 20 * np.log10(max(float(np.sqrt((w[:n] ** 2).mean())), 1e-9) / peak)
    tail = 20 * np.log10(max(float(np.sqrt((w[-n:] ** 2).mean())), 1e-9) / peak)
    print(f"  {name:34} {len(w)/sr:5.2f}s   head {head:6.1f} dB   tail {tail:6.1f} dB")
    return path


VARIANTS = [
    ("1_current_pipeline",        base,        True,  True),
    ("2_no_trim_no_fade",         base,        False, False),
    ("3_duration_x1.4",           base * 1.4,  True,  True),
    ("4_duration_x1.4_raw",       base * 1.4,  False, False),
    ("5_duration_x1.8_raw",       base * 1.8,  False, False),
    ("6_f5tts_own_estimate_raw",  None,        False, False),
]

print(f"  {'variant':34} {'len':>7}   {'head':>10}   {'tail':>10}")
for name, fix, do_trim, do_fade in VARIANTS:
    w, sr = render(fix)
    if do_trim:
        w = app.trim_silence(w, sr)
    if do_fade:
        w = app.fade_edges(w, sr)
    report(name, w, sr)

print(f"\nwritten to {out_dir}")
print("\nListen in order. The first one where the last word is COMPLETE tells")
print("you what was eating it:")
print("  2 complete, 1 not  -> our trim/fade is the culprit")
print("  3 or 4 complete    -> the duration budget was too tight")
print("  5 complete         -> needs much more room than estimated")
print("  none complete      -> the model stops early regardless; that is a")
print("                        model limit, not a setting")
