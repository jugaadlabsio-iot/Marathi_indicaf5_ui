# -*- coding: utf-8 -*-
"""
Render one full story twice, once per reference clip, so the choice is made on
a whole story rather than a six-line passage.

    python tools/story_refs.py --story 01_hunted_well --refs ref_nat3 ref_nat1

01_hunted_well is the worst case in the corpus: 89% of its paragraphs are short
or dialogue (80 chars each), and chaining engaged on only 45% of its chunks
before the paragraph fix. If a reference holds up here it holds up anywhere.

Settings are production: seed 7 locked, NFE 32, CFG 2.0, roominess 1.35,
max 8.0s per chunk, anchored chaining across paragraphs, warmth on.

Waits for any other render to finish first - two jobs on one throttled GTX 1650
take longer than the same two run back to back.
"""
import argparse
import ctypes
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
p.add_argument("--story", default="01_hunted_well")
p.add_argument("--refs", nargs="+", default=["ref_nat3", "ref_nat1"])
p.add_argument("--after-pid", type=int, default=0,
               help="wait for this render to finish before starting")
a = p.parse_args()


def alive(pid):
    if not pid:
        return False
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, pid)
    if not h:
        return False
    try:
        c = ctypes.c_ulong()
        return bool(k.GetExitCodeProcess(h, ctypes.byref(c))) and c.value == 259
    finally:
        k.CloseHandle(h)


if a.after_pid:
    print(f"waiting for pid {a.after_pid} to finish first", flush=True)
    while alive(a.after_pid):
        time.sleep(30)
    print(f"pid {a.after_pid} done at {time.strftime('%H:%M')}", flush=True)

spec = importlib.util.spec_from_file_location("mtts_app", HERE / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

# A degraded render of a whole story is 2+ hours wasted. Check before starting.
for mod, name in ((app.pacing, "pacing"), (app.numerals, "numerals"),
                  (app.warmth, "warmth")):
    if mod is None:
        sys.exit(f"app.py could not import {name} - refusing to start a "
                 f"multi-hour render that would be silently degraded.")
print("modules ok: pacing, numerals, warmth\n", flush=True)

src = HERE / "queue" / "done" / f"{a.story}.txt"
if not src.exists():
    sys.exit(f"no such story: {src}")
script = src.read_text(encoding="utf-8")

ckpt = next(c for c in app.list_ckpts() if "voice_merged" in c.lower())
d = app.load_dict()
pauses = {"chunk": 250, "flow": 60, "sent": 300, "line": 350,
          "spara": 320, "para": 600, "end": 0}

print(f"story : {a.story}  ({len(script)} chars)")
print(f"model : {Path(ckpt).name}")
print(f"refs  : {', '.join(a.refs)}\n", flush=True)

for r in a.refs:
    refwav = HERE / "ref" / f"{r}.wav"
    if not refwav.exists():
        print(f"  {r}: missing, skipped", flush=True)
        continue
    t0 = time.time()
    print(f"[{r}] starting {time.strftime('%H:%M')}", flush=True)
    try:
        out, dur, took, n, run_dir = app.synthesize(
            script, ckpt, str(refwav), app.ref_text_for(str(refwav)),
            1.0, 32, 400, pauses,
            True, [[k, v] for k, v in d.items()], True, app.DEVICE,
            out_name=f"{a.story}__{r}",
            on_progress=lambda f, m: print(f"    {m}", end="\r", flush=True),
            log=lambda m: print(m, flush=True),
            # Everything stated explicitly. These are all currently the
            # defaults, but a render that costs two hours should not depend on
            # a default staying put, and the run_info.txt record is only worth
            # anything if the call it describes is unambiguous.
            cfg=2.0, sway=-1.0, pace=1.35, seed=7, seed_per_chunk=False,
            max_secs=8.0, lead_ms=350, tail_ms=900,
            trim=True,                 # strip the model's ragged head/tail
            drop_directions=True,      # do not narrate [stage directions]
            expand_nums=True,          # digits -> Marathi words
            one_pass=True,             # never let F5-TTS split and cross-fade
            fit_duration=True,         # syllable budgeting, not byte count
            per_sentence=True,
            retry_short=True,
            chain=True, chain_reanchor=8, chain_anchored=True,
            drift_limit=0.045, chain_across_paragraphs=True,
            apply_warmth=True)
        rows = (run_dir / "chunks.tsv").read_text(encoding="utf-8").splitlines()[1:]
        budgets = [float(x.split("\t")[2]) for x in rows if x.strip()]
        flag = "budgets OK" if all(b > 0 for b in budgets) else "*** UNPACED ***"
        print(f"\n[{r}] {out.name}  {dur/60:.1f} min audio, {n} chunks, "
              f"{(time.time()-t0)/60:.0f} min  {flag}", flush=True)
    except Exception as e:
        print(f"\n[{r}] FAILED: {e}", flush=True)

print("\nBoth renders written to out/ with _raw.wav alongside each.", flush=True)
