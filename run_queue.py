# -*- coding: utf-8 -*-
"""
Unattended overnight queue runner - no browser needed.

    python run_queue.py                 # sensible defaults
    python run_queue.py --nfe 32        # final-quality pass
    python run_queue.py --device cpu    # if the GPU misbehaves

Reads every *.txt in queue/, writes a wav per story into out/, and moves each
source into queue/done/ or queue/failed/.
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mtts_app", HERE / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)          # safe: launch() is behind __main__

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=None, help="defaults to the slim checkpoint if present")
p.add_argument("--ref", default=None, help="defaults to the shortest reference clip")
p.add_argument("--nfe", type=int, default=32)
p.add_argument("--speed", type=float, default=1.0)
p.add_argument("--max-chars", type=int, default=400)
p.add_argument("--pause", type=int, default=250)
p.add_argument("--line-pause", type=int, default=350)
p.add_argument("--para-pause", type=int, default=800)
p.add_argument("--device", default="auto")
p.add_argument("--cfg", type=float, default=2.0, help="CFG strength (2.0 balanced)")
p.add_argument("--no-trim", action="store_true", help="keep the model's own head/tail silence")
p.add_argument("--keep-directions", action="store_true", help="read [stage directions] aloud")
p.add_argument("--keep-digits", action="store_true",
               help="do NOT spell numbers out as Marathi words")
p.add_argument("--allow-splits", action="store_true",
               help="let F5-TTS split long chunks itself (its seams cause slurring)")
p.add_argument("--sent-pause", type=int, default=300, help="pause at a full stop (ms)")
p.add_argument("--pace", type=float, default=1.35,
               help="roominess. 1.35 is the tested default: the same sentence "
                    "on 10 random seeds was clean 4/10 at 1.0 and 9/10 here. "
                    "A starved chunk rushes whichever word is hardest, and "
                    "which word that is changes with the seed.")
p.add_argument("--byte-duration", action="store_true",
               help="use F5-TTS's byte-count duration guess instead of syllables")
p.add_argument("--no-sentence-split", action="store_true",
               help="pack several sentences per chunk (they will run together)")
p.add_argument("--seed", type=int, default=1097657232,
               help="fixed so a run is reproducible and a bad chunk can be "
                    "re-rendered deliberately; F5-TTS otherwise draws a fresh "
                    "random seed per chunk")
p.add_argument("--lead", type=int, default=350, help="silence before the first word (ms)")
p.add_argument("--tail", type=int, default=900, help="silence after the last word (ms)")
p.add_argument("--max-secs", type=float, default=3.2,
               help="split so no chunk exceeds this many seconds (0 = no cap); "
                    "long chunks slur the retroflexes")
a = p.parse_args()

ckpts = app.list_ckpts()
if not ckpts:
    sys.exit(f"No checkpoints in {app.MODELS_DIR}")
ckpt = a.ckpt or next((c for c in ckpts if "slim" in c.lower()), ckpts[0])

refs = app.list_refs()
if not refs:
    sys.exit(f"No reference clips in {app.REF_DIR}")
if a.ref:
    ref = a.ref
else:                                  # shortest clip = fastest generation
    import soundfile as sf
    ref = min(refs, key=lambda r: sf.info(r).duration)
ref_txt = app.ref_text_for(ref)
if not ref_txt:
    sys.exit(f"No transcript beside {ref} - create a matching .txt")

device = app.DEVICE if a.device == "auto" else a.device
pauses = {"chunk": a.pause, "sent": a.sent_pause, "line": a.line_pause,
          "para": a.para_pause, "end": 0}
d = app.load_dict()

jobs = app.list_queue()
print("=" * 60)
print(f"queue    : {len(jobs)} story(ies)")
print(f"device   : {device}")
print(f"model    : {Path(ckpt).name}")
print(f"reference: {Path(ref).name}")
print(f"NFE      : {a.nfe}")
print("=" * 60, flush=True)
if not jobs:
    sys.exit(0)

t_all = time.time()
ok = fail = 0
for i, name in enumerate(jobs, 1):
    src = app.QUEUE_DIR / name
    if not src.exists():
        continue
    title = src.stem
    print(f"\n[{i}/{len(jobs)}] {title}", flush=True)
    try:
        out_path, dur, took, n, _ = app.synthesize(
            src.read_text(encoding="utf-8"), ckpt, ref, ref_txt,
            a.speed, a.nfe, a.max_chars, pauses,
            True, [[k, v] for k, v in d.items()], True, device,
            out_name=title,
            on_progress=lambda f, msg: print(f"    {msg}", end="\r", flush=True),
            log=lambda m: print(m, flush=True),
            cfg=a.cfg, trim=not a.no_trim,
            drop_directions=not a.keep_directions,
            expand_nums=not a.keep_digits,
            one_pass=not a.allow_splits,
            pace=a.pace, fit_duration=not a.byte_duration,
            per_sentence=not a.no_sentence_split, seed=a.seed,
            max_secs=a.max_secs, lead_ms=a.lead, tail_ms=a.tail)
        import shutil
        shutil.move(str(src), str(app.QUEUE_DONE / name))
        ok += 1
        print(f"\n    OK  {dur/60:.1f} min audio · {n} chunks · "
              f"{took/60:.1f} min -> {out_path.name}", flush=True)
    except Exception as e:
        import shutil
        shutil.move(str(src), str(app.QUEUE_FAILED / name))
        fail += 1
        print(f"\n    FAILED: {e}", flush=True)

print("\n" + "=" * 60)
print(f"done: {ok} ok, {fail} failed, {(time.time()-t_all)/60:.1f} min total")
print(f"audio in {app.OUT_DIR}")
print("=" * 60)
