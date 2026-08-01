# -*- coding: utf-8 -*-
"""Re-render individual bad chunks and rebuild the story around them.

Even at the tested roominess about one chunk in ten still lands badly, and a
two-hour rerun to fix thirty seconds of audio is a poor trade. Every chunk was
written to disk as it was made, so a repair only needs to redo the ones you
name.

    # what does chunk 47 actually say?
    python tools/repair.py --run 01_hunted_well_20260801-113512 --show 40-60

    # re-render some chunks and rebuild the wav
    python tools/repair.py --run 01_hunted_well_20260801-113512 --fix 47,52,88

    # same, but give those chunks even more time
    python tools/repair.py --run ... --fix 47 --pace 1.35 --tries 3

    # rebuild only, after you have replaced a chunk wav by hand
    python tools/repair.py --run ... --rebuild

`--tries N` renders N candidates on different seeds and keeps the longest
clean one, which in practice means the least rushed. The original chunk is
kept as NNNN.wav.orig so a repair can always be undone.
"""
import argparse
import shutil
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


def parse_list(s):
    """'47,52,88' or '40-60' or a mix."""
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


p = argparse.ArgumentParser()
p.add_argument("--run", required=True, help="folder name under out/parts/")
p.add_argument("--show", default=None, help="print the text of these chunks")
p.add_argument("--fix", default=None, help="chunk numbers to re-render")
p.add_argument("--rebuild", action="store_true", help="reassemble without re-rendering")
p.add_argument("--tries", type=int, default=3, help="candidates per chunk")
p.add_argument("--pace", type=float, default=1.35)
p.add_argument("--nfe", type=int, default=32)
p.add_argument("--cfg", type=float, default=2.0)
p.add_argument("--pause", type=int, default=250)
p.add_argument("--sent-pause", type=int, default=300)
p.add_argument("--line-pause", type=int, default=350)
p.add_argument("--para-pause", type=int, default=800)
p.add_argument("--lead", type=int, default=350)
p.add_argument("--tail", type=int, default=900)
p.add_argument("--restore", default=None, help="undo repairs on these chunks")
a = p.parse_args()

run_dir = HERE / "out" / "parts" / a.run
if not run_dir.is_dir():
    sys.exit(f"no such run: {run_dir}")
index = run_dir / "chunks.tsv"
if not index.exists():
    sys.exit(f"{index} is missing - this run predates chunk indexing, so its "
             f"chunks cannot be mapped back to text. Re-runs from now on have it.")

rows = []
for line in index.read_text(encoding="utf-8").splitlines()[1:]:
    f, kind, planned, text = (line.split("\t", 3) + ["", "", "", ""])[:4]
    rows.append(dict(file=f, kind=kind, planned=float(planned or 0), text=text))
print(f"{a.run}: {len(rows)} chunks")

if a.show:
    for i in parse_list(a.show):
        if 1 <= i <= len(rows):
            r = rows[i - 1]
            w = run_dir / r["file"]
            dur = sf.info(w).duration if w.exists() else 0.0
            print(f"  {i:4d} [{r['kind']:5}] planned {r['planned']:4.1f}s "
                  f"actual {dur:4.1f}s  {r['text']}")
    if not (a.fix or a.rebuild or a.restore):
        sys.exit(0)

if a.restore:
    n = 0
    for i in parse_list(a.restore):
        w = run_dir / rows[i - 1]["file"]
        orig = w.with_suffix(w.suffix + ".orig")
        if orig.exists():
            shutil.copy(orig, w)
            n += 1
    print(f"restored {n} chunk(s) from .orig")

if a.fix:
    ckpt = str(app.MODELS_DIR / "model_last_slim.pt")
    ref = min(app.list_refs(), key=lambda r: sf.info(r).duration)
    ref_txt = app.ref_text_for(ref)
    _, ref_norm, ref_sec = app.ref_profile(ref, ref_txt.strip())
    rate = pacing.speech_rate(ref_norm, ref_sec)
    tts = app.get_tts(ckpt, str(app.MODELS_DIR / "vocab.txt"), app.DEVICE)
    room = min(app.MAX_ROOMINESS, 1.08 * a.pace)

    for i in parse_list(a.fix):
        if not (1 <= i <= len(rows)):
            print(f"  {i}: out of range")
            continue
        r = rows[i - 1]
        w = run_dir / r["file"]
        orig = w.with_suffix(w.suffix + ".orig")
        if w.exists() and not orig.exists():
            shutil.copy(w, orig)          # keep the original, always
        want = pacing.estimate_seconds(r["text"], rate, roominess=room)
        print(f"\n  chunk {i}: {r['text'][:60]}")
        best, best_len, sr = None, -1.0, 24000
        for t in range(a.tries):
            sd = 1097657232 + i + t * 7919      # deterministic, different each try
            wav, sr, _ = tts.infer(
                gen_text=r["text"], ref_file=ref, ref_text=ref_txt.strip(),
                speed=1.0, nfe_step=a.nfe, cfg_strength=a.cfg,
                sway_sampling_coef=-1.0, remove_silence=False,
                seed=sd, fix_duration=ref_sec + want)
            wav = np.asarray(wav, dtype=np.float32)
            if np.isnan(wav).any():
                continue
            wav = app.trim_silence(wav, sr)
            bad = app.ends_mid_word(wav, sr) or app.starts_mid_word(wav, sr)
            # longest clean take = the one that rushed its words least
            score = -1.0 if bad else len(wav) / sr
            print(f"    try {t+1}: seed {sd}  {len(wav)/sr:4.2f}s"
                  f"{'  clipped, skipped' if bad else ''}")
            if score > best_len:
                best, best_len = wav, score
        if best is None:
            print("    all tries failed, leaving the original in place")
            continue
        sf.write(w, app.fade_edges(best, sr), sr)
        print(f"    kept {best_len:4.2f}s -> {w.name}")

# rebuild the story from whatever chunk wavs are on disk now
pauses = {"chunk": a.pause, "sent": a.sent_pause, "line": a.line_pause,
          "para": a.para_pause, "end": 0}
pieces, sr = [], 24000
for n, r in enumerate(rows, 1):
    w = run_dir / r["file"]
    if not w.exists():
        continue
    x, sr = sf.read(w)
    pieces.append(np.asarray(x, dtype=np.float32))
    gap = pauses.get(r["kind"], 0)
    if gap and n < len(rows):
        pieces.append(np.zeros(int(sr * gap / 1000), dtype=np.float32))

audio = np.concatenate([
    np.zeros(int(sr * a.lead / 1000), dtype=np.float32),
    np.concatenate(pieces),
    np.zeros(int(sr * a.tail / 1000), dtype=np.float32)])
audio = audio / (np.abs(audio).max() or 1.0) * 0.95

out = app.OUT_DIR / f"{a.run}_repaired.wav"
sf.write(out, audio, sr)
print(f"\nrebuilt: {out}   {len(audio)/sr/60:.1f} min from {len(pieces)} pieces")
