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

# Windows consoles default to cp1252, which cannot encode Devanagari - every
# story then dies with "'charmap' codec can't encode characters". Do not rely
# on the caller exporting PYTHONIOENCODING.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
p.add_argument("--flow-pause", type=int, default=60,
               help="pause between pieces of ONE sentence. Small on purpose: "
                    "a split sentence should read as a breath, not a stop")
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
p.add_argument("--cool-below", type=int, default=0,
               help="between stories, wait until the GPU drops to this "
                    "temperature. Sustained load pins the card at 90C where it "
                    "thermally throttles; renders measured 22s cold and 65s hot, "
                    "so cooling between stories can cost less time than it saves")
p.add_argument("--cool-max-min", type=int, default=25,
               help="give up waiting for the GPU after this many minutes")
p.add_argument("--no-chain", action="store_true",
               help="do NOT condition each chunk on the previous one. Chaining is "
                    "on by default: F5-TTS generates a CONTINUATION of whatever "
                    "audio it is given, so feeding it the previous chunk carries "
                    "intonation across the join instead of restarting it from the "
                    "reference clip 200 times a story")
p.add_argument("--chain-reanchor", type=int, default=6,
               help="go back to the real reference clip every N chunks, and at "
                    "every paragraph, so synthetic-on-synthetic drift stays bounded")
p.add_argument("--no-warmth", action="store_true",
               help="skip the post-vocoder EQ and compression. On by default; "
                    "the unprocessed mix is always kept as <name>_raw.wav")
p.add_argument("--reverse", action="store_true",
               help="work through the queue bottom-up, so a story you have "
                    "not heard yet finishes first and can be judged early")
p.add_argument("--lock-seed", action="store_true",
               help="use the SAME seed for every chunk instead of seed+i; "
                    "the probes did this and came out consistently clean")
p.add_argument("--lead", type=int, default=350, help="silence before the first word (ms)")
p.add_argument("--tail", type=int, default=900, help="silence after the last word (ms)")
p.add_argument("--max-secs", type=float, default=3.2,
               help="split so no chunk exceeds this many seconds (0 = no cap); "
                    "long chunks slur the retroflexes")
a = p.parse_args()

ckpts = app.list_ckpts()
if not ckpts:
    sys.exit(f"No checkpoints in {app.MODELS_DIR}")
# The merged checkpoint is the default. The pure fine-tune had overwritten
# the base model's Marathi - blending it half-way back recovered correct
# pronunciation while keeping the voice recognisable.
ckpt = a.ckpt or next((c for c in ckpts if "voice_merged" in c.lower()),
                      next((c for c in ckpts if "slim" in c.lower()), ckpts[0]))

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

def gpu_temp():
    """Current GPU temperature, or None if nvidia-smi is not usable."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def cool_down(label=""):
    """Pause until the card is cool enough to run at full clocks."""
    if a.cool_below <= 0:
        return
    t = gpu_temp()
    if t is None or t <= a.cool_below:
        return
    print(f"\n  cooling {label}: GPU at {t}C, waiting for {a.cool_below}C "
          f"(up to {a.cool_max_min} min)", flush=True)
    start = time.time()
    while True:
        time.sleep(30)
        t = gpu_temp()
        mins = (time.time() - start) / 60
        if t is None or t <= a.cool_below:
            print(f"  cooled to {t}C after {mins:.0f} min", flush=True)
            return
        if mins >= a.cool_max_min:
            print(f"  still {t}C after {mins:.0f} min, carrying on", flush=True)
            return


STOP_FILE = app.QUEUE_DIR / "STOP"


def stop_requested():
    """Has someone asked for a graceful stop?

    Checked between stories, never during one, so the story being rendered
    always finishes. Dropping a file is the only mechanism that works from
    another terminal, another machine over a share, or a script - and unlike
    watching the log it cannot misread a missing command as a dead process.
    """
    return STOP_FILE.exists()


def clear_stop():
    try:
        STOP_FILE.unlink()
    except OSError:
        pass


device = app.DEVICE if a.device == "auto" else a.device
pauses = {"chunk": a.pause, "flow": a.flow_pause, "sent": a.sent_pause, "line": a.line_pause,
          "para": a.para_pause, "end": 0}
d = app.load_dict()

jobs = app.list_queue()
if a.reverse:
    jobs = list(reversed(jobs))
print("=" * 60)
print(f"queue    : {len(jobs)} story(ies)")
print(f"device   : {device}")
print(f"model    : {Path(ckpt).name}")
print(f"reference: {Path(ref).name}")
print(f"NFE      : {a.nfe}")
print("=" * 60, flush=True)
if not jobs:
    sys.exit(0)
if stop_requested():
    print(f"{STOP_FILE} is present - nothing started. Delete it to run.", flush=True)
    clear_stop()
    sys.exit(0)

t_all = time.time()
ok = fail = 0
for i, name in enumerate(jobs, 1):
    src = app.QUEUE_DIR / name
    if not src.exists():
        continue
    title = src.stem
    if stop_requested():
        print(f"\n  stop requested - finishing here, {len(jobs) - i + 1} "
              f"story(ies) left in the queue", flush=True)
        clear_stop()
        break
    cool_down(f"before {title}")
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
            max_secs=a.max_secs, lead_ms=a.lead, tail_ms=a.tail,
            seed_per_chunk=not a.lock_seed,
            chain=not a.no_chain, chain_reanchor=a.chain_reanchor,
            apply_warmth=not a.no_warmth)
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
