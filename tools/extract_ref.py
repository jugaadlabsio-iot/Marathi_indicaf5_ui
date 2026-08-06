# -*- coding: utf-8 -*-
"""
Cut a reference clip out of your own master recordings - no time-stretching.

    python tools/extract_ref.py --list
    python tools/extract_ref.py --source "Vihiritla aavaj" --story 01_hunted_well

Why
---
Everything inherits the reference clip's articulation rate, because F5-TTS
generates a continuation of whatever it conditions on. ref_short.wav runs at
9.57 moras/sec. Measured across seven full narrations on D:, YOUR natural
narration averages 8.06 - so the clip driving every story is 19% faster than
you actually read.

app.py's note calls ~6 moras/sec "narration pace". That is a generic figure and
it is wrong for this voice: time-stretched clips at 6.07 and 5.38 were rejected
as both too slow AND degraded. The target here is YOUR measured 8.06, reached by
cutting real audio rather than stretching it, so there are no phase-vocoder
artifacts to trade against.

Transcripts
-----------
Whisper's Marathi is not reliable enough to transcribe a reference (it renders
वाजता as वास्ता and splits लॅपटॉपवर in two). But the story text is on disk, so
Whisper only has to be good enough to FIND which sentence a segment is - the
transcript written out is the exact sentence from your script. A candidate whose
match falls below --min-match is discarded rather than written with text that
does not correspond to its audio.
"""
import argparse
import difflib
import re
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import numpy as np
import soundfile as sf
import librosa
import pacing

SRC = Path("D:/Audio Recordings")
DONE = HERE / "queue" / "done"

p = argparse.ArgumentParser()
p.add_argument("--list", action="store_true")
p.add_argument("--source", default="Vihiritla aavaj")
p.add_argument("--story", default="01_hunted_well")
p.add_argument("--target-rate", type=float, default=8.06,
               help="your measured natural narration rate")
p.add_argument("--min-secs", type=float, default=4.0)
p.add_argument("--max-secs", type=float, default=6.0)
p.add_argument("--scan-minutes", type=float, default=6.0,
               help="how much of the master to search")
p.add_argument("--skip-minutes", type=float, default=0.5,
               help="skip the opening, which is usually title/intro")
p.add_argument("--pad", type=float, default=0.18,
               help="seconds of silence to keep at BOTH edges. A reference that "
                    "starts or ends mid-word teaches the model to clip")
p.add_argument("--min-match", type=float, default=0.65,
               help="how well Whisper must agree with the story sentence. "
                    "Low scores mean the transcript may not correspond to "
                    "the audio, which is the one thing a reference must "
                    "never get wrong")
p.add_argument("--top", type=int, default=4)
p.add_argument("--model", default="small")
a = p.parse_args()

if a.list:
    for f in sorted(SRC.glob("*.wav")):
        print(f"  {f.stem:<44}{sf.info(str(f)).duration/60:5.1f} min")
    sys.exit(0)

src = SRC / f"{a.source}.wav"
if not src.exists():
    sys.exit(f"no such recording: {src}")
story = DONE / f"{a.story}.txt"
if not story.exists():
    sys.exit(f"no such story text: {story}")

# --- load the scan window as 24k mono, matching ref_short.wav ---------------
info = sf.info(str(src))
start = int(a.skip_minutes * 60 * info.samplerate)
frames = int(a.scan_minutes * 60 * info.samplerate)
x, sr = sf.read(str(src), start=start, frames=frames, dtype="float32", always_2d=True)
x = x.mean(axis=1)
if sr != 24000:
    x = librosa.resample(x, orig_sr=sr, target_sr=24000)
    sr = 24000
print(f"scanning {a.scan_minutes:.0f} min of {src.name} from {a.skip_minutes:.1f} min\n")

# --- silence-bounded segments ----------------------------------------------
win = int(sr * 0.02)
n = len(x) // win
db = 20 * np.log10(np.maximum(np.sqrt((x[:n * win].reshape(n, win) ** 2).mean(1)), 1e-9))
quiet = db < -38.0

# Keep whole quiet RUNS, not their midpoints, so a segment can be cut to begin
# and end inside silence. A reference that starts or ends mid-word is exactly
# what teaches the model to clip: it conditions on a truncated onset and
# reproduces one. Cutting inside the pause guarantees clean edges by
# construction rather than by luck.
runs, run = [], None
for i, q in enumerate(quiet):
    if q and run is None:
        run = i
    elif not q and run is not None:
        if (i - run) * 0.02 >= 0.30:
            runs.append((run * 0.02, i * 0.02))
        run = None
if run is not None and (len(quiet) - run) * 0.02 >= 0.30:
    runs.append((run * 0.02, len(quiet) * 0.02))
print(f"found {len(runs)} pauses >= 300ms")

cands = []
for i in range(len(runs) - 1):
    lead_room = runs[i][1] - runs[i][0]
    if lead_room < a.pad:
        continue
    s0 = runs[i][1] - a.pad                      # `pad` of silence, then speech
    for j in range(i + 1, min(i + 6, len(runs))):
        if runs[j][1] - runs[j][0] < a.pad:
            continue
        s1 = runs[j][0] + a.pad                  # speech, then `pad` of silence
        if a.min_secs <= s1 - s0 <= a.max_secs:
            cands.append((s0, s1))
print(f"{len(cands)} candidate segments of {a.min_secs:.0f}-{a.max_secs:.0f}s "
      f"with >= {a.pad*1000:.0f}ms silence at both edges")
if not cands:
    sys.exit("none - widen --min-secs/--max-secs or --scan-minutes")

# --- the story's sentences are the source of truth for text ----------------
body = "\n".join(l for l in story.read_text(encoding="utf-8", errors="replace").splitlines()
                 if not l.strip().startswith(("[", "*", "#")))
# One clean sentence per candidate. A reference spanning a line break or a
# dialogue boundary carries unbalanced quotes and two speakers' worth of
# prosody - both showed up in the first pass ("सांगतो एवढं ऐका." + "मला
# सरकारी काम आहे) and neither belongs in a conditioning clip.
sents = []
for s_ in re.split(r"(?<=[।\.\?\!])\s+", body):
    s_ = re.sub(r"\s+", " ", s_).strip()
    if len(s_) <= 12:
        continue
    if s_.count('"') % 2 or s_.count("“") != s_.count("”"):
        continue                                  # unbalanced quotes
    if s_[0] in '"“‘' or s_[-1] in '“‘':
        continue                                  # opens mid-dialogue
    sents.append(s_)
print(f"{len(sents)} sentences in {a.story}\n")


def norm(s):
    return re.sub(r"[^ऀ-ॿ]", "", s)


nsents = [norm(s) for s in sents]

from faster_whisper import WhisperModel
print(f"transcribing candidates (faster-whisper '{a.model}', CPU)...")
model = WhisperModel(a.model, device="cpu", compute_type="int8")

tmp = HERE / "out" / "_extract_tmp.wav"
tmp.parent.mkdir(parents=True, exist_ok=True)
scored = []
for k, (s0, s1) in enumerate(cands):
    seg = x[int(s0 * sr):int(s1 * sr)]
    sf.write(str(tmp), seg, sr)
    try:
        segs, _ = model.transcribe(str(tmp), language="mr")
        heard = norm(" ".join(t.text for t in segs))
    except Exception:
        continue
    if len(heard) < 8:
        continue
    best_i, best_r = -1, 0.0
    for i, ns in enumerate(nsents):
        r = difflib.SequenceMatcher(None, heard, ns, autojunk=False).ratio()
        if r > best_r:
            best_i, best_r = i, r
    if best_r < a.min_match:
        continue
    frames_ = len(seg) // win
    sdb = 20 * np.log10(np.maximum(
        np.sqrt((seg[:frames_ * win].reshape(frames_, win) ** 2).mean(1)), 1e-9))
    speech = (sdb >= -38.0).sum() * 0.02
    text = sents[best_i]
    rate = pacing.moras(text) / max(speech, 1e-6)
    # Trust nothing: measure the edges of the audio actually cut.
    edge = int(0.12 * sr)
    lead_db = 20 * np.log10(max(float(np.sqrt((seg[:edge] ** 2).mean())), 1e-9))
    tail_db = 20 * np.log10(max(float(np.sqrt((seg[-edge:] ** 2).mean())), 1e-9))
    if lead_db > -38.0 or tail_db > -38.0:
        continue                                  # an edge landed on speech
    scored.append(dict(s0=s0, s1=s1, dur=s1 - s0, text=text, match=best_r,
                       rate=rate, peak=float(np.max(np.abs(seg))),
                       lead_db=lead_db, tail_db=tail_db))

try:
    tmp.unlink()
except OSError:
    pass

if len(scored) < 2:
    sys.exit(f"only {len(scored)} clean candidate(s) - need at least 2.\n"
             f"Try --min-match 0.35, a longer --scan-minutes, or --pad 0.12")

scored.sort(key=lambda c: abs(c["rate"] - a.target_rate))
print(f"\n{len(scored)} matched. Closest to your natural {a.target_rate:.2f} moras/sec:\n")
REF = HERE / "ref"
for i, c in enumerate(scored[:a.top], 1):
    name = f"ref_nat{i}"
    seg = x[int(c["s0"] * sr):int(c["s1"] * sr)]
    seg = seg / (np.max(np.abs(seg)) or 1.0) * 0.95
    sf.write(str(REF / f"{name}.wav"), seg.astype(np.float32), sr, subtype="PCM_16")
    (REF / f"{name}.txt").write_text(c["text"], encoding="utf-8")
    print(f"  {name}.wav  {c['dur']:4.2f}s  {c['rate']:5.2f} m/s  "
          f"match {c['match']:.2f}  headroom {11.0 - c['dur']:4.1f}s  "
          f"edges {c['lead_db']:.0f}/{c['tail_db']:.0f} dB")
    print(f"      {c['text'][:88]}")

print("\nThese are cut from your own voice - no stretching, no artifacts.")
print("Audition them, then set the winner in run_queue.py with --ref.")
