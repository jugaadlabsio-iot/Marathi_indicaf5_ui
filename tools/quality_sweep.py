# -*- coding: utf-8 -*-
"""
NFE and CFG have never been varied. Vary them.

    python tools/quality_sweep.py

Everything shipped so far has used nfe=32, cfg=2.0, sway=-1.0 - the values
picked on day one and never revisited. They are the largest untested part of
the parameter space, and in a flow-matching model NFE is the direct
quality/time trade: it is the number of ODE solver steps between noise and
speech.

  NFE   more steps = a more faithful solution to the same ODE. 32 is the
        F5-TTS default; 48-64 is where diminishing returns usually set in.
        Cost is linear - 64 takes twice as long as 32.
  CFG   classifier-free guidance. Higher sticks closer to the text and can
        sound clipped and tense; lower is looser and more natural but can
        slur or drift off the script.
  sway  sway sampling shifts where the solver spends its steps. -1.0 front-
        loads them, which the F5-TTS authors found better for speech.

Renders one passage per combination so the difference is audible. This is
deliberately a small grid: on a throttled GTX 1650 each render is minutes,
and a 30-cell grid nobody listens to is worth less than 6 cells you compare
properly.
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

p = argparse.ArgumentParser()
p.add_argument("--grid", default="quality",
               choices=["quality", "cfg", "sway"],
               help="quality: NFE ladder at the current CFG. "
                    "cfg: CFG ladder at the best NFE. sway: sway ladder.")
p.add_argument("--nfe", type=int, default=32, help="fixed NFE for the cfg/sway grids")
a = p.parse_args()

spec = importlib.util.spec_from_file_location("mtts_app", HERE / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
for m, n in ((app.pacing, "pacing"), (app.numerals, "numerals"), (app.warmth, "warmth")):
    if m is None:
        sys.exit(f"app.py could not import {n} - refusing to run a comparison "
                 f"that would be silently degraded.")

# A passage with narration, dialogue and a question - the three things that
# have each failed differently at some point.
PASSAGE = """गणपतरावांनी जीपमधून उतरून सभोवताली नजर फिरवली. गाव शांत होतं.

"इथे कोणी राहतं का?"

म्हातारा हसला. त्याचे डोळे बारीक झाले.

"राहतं. पण दिसत नाही."
"""

GRIDS = {
    # (label, nfe, cfg, sway)
    "quality": [("nfe32_baseline", 32, 2.0, -1.0),
                ("nfe48", 48, 2.0, -1.0),
                ("nfe64", 64, 2.0, -1.0)],
    "cfg":     [(f"cfg1.5_nfe{a.nfe}", a.nfe, 1.5, -1.0),
                (f"cfg2.0_nfe{a.nfe}", a.nfe, 2.0, -1.0),
                (f"cfg2.5_nfe{a.nfe}", a.nfe, 2.5, -1.0)],
    "sway":    [(f"sway-1.0_nfe{a.nfe}", a.nfe, 2.0, -1.0),
                (f"sway-0.5_nfe{a.nfe}", a.nfe, 2.0, -0.5),
                (f"sway0.0_nfe{a.nfe}", a.nfe, 2.0, 0.0)],
}

ckpt = next(c for c in app.list_ckpts() if "voice_merged" in c.lower())
ref = app.default_ref()
d = app.load_dict()
pauses = {"chunk": 250, "flow": 60, "sent": 300, "line": 350,
          "spara": 320, "para": 600, "end": 0}

print(f"model     {Path(ckpt).name}")
print(f"reference {Path(ref).name}")
print(f"grid      {a.grid}\n", flush=True)

for label, nfe, cfg, sway in GRIDS[a.grid]:
    t0 = time.time()
    try:
        out, dur, took, n, rd = app.synthesize(
            PASSAGE, ckpt, ref, app.ref_text_for(ref),
            1.0, nfe, 400, pauses,
            True, [[k, v] for k, v in d.items()], True, app.DEVICE,
            out_name=f"_q_{label}",
            on_progress=lambda f, m: None, log=lambda m: None,
            cfg=cfg, sway=sway, pace=1.35, seed=7, seed_per_chunk=False,
            max_secs=8.0, lead_ms=300, tail_ms=700, edge_retries=3,
            chain=True, chain_reanchor=8, chain_across_paragraphs=True,
            apply_warmth=True)
        info = (rd / "run_info.txt").read_text(encoding="utf-8")
        clipped = next((l.split()[1] for l in info.splitlines()
                        if l.startswith("CLIPPED")), "?")
        print(f"  {label:<20} {dur:5.1f}s  {n} chunks  {time.time()-t0:5.0f}s  "
              f"clipped={clipped}", flush=True)
    except Exception as e:
        print(f"  {label:<20} FAILED: {e}", flush=True)

print("\nListen for: consonant crispness, sibilance, and whether the question")
print("line keeps its rising contour. Higher NFE should mostly clean up the")
print("texture rather than change the delivery.")
