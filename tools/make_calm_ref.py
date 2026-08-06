# -*- coding: utf-8 -*-
"""
Build a reference clip that is BOTH calm and short.

    python tools/make_calm_ref.py
    python tools/make_calm_ref.py --show-asr     # just print the alignment

The problem
-----------
Everything inherits the reference clip's articulation rate, because F5-TTS
generates a continuation of whatever it is conditioned on. Measured:

    ref_short.wav   9.57 moras/sec of actual speech
    narration       ~6

(ref_9s.wav looks calmer at 6.62, but that is a clip average inflated by 2.24s
of internal silence. Its articulation rate is 8.82 - the same delivery with
longer pauses - so cutting it up gains nothing.)

Time-stretching fixes the rate but lengthens the clip, and anchored chaining
needs 11.0 - ref_seconds of headroom for the previous chunk. At a 5.5s mean
chunk an 8.2s calm clip leaves 2.8s and chaining stops happening, so calm and
flowing would trade against each other. Hence: stretch, then truncate at a word
boundary and write the matching transcript prefix.

The transcript always comes from YOUR text file, never from the ASR - Whisper's
Marathi is not reliable enough to be the source of truth. Whisper supplies only
timings, aligned to your words with difflib so that a single split or merged
token does not invalidate the run. Anything that cannot be aligned is refused
rather than written as a mismatched pair: ref_text has to correspond to
ref_audio, or the duration budget is computed against words that are not there.

Word timings are taken from the ORIGINAL clip and scaled. Time-stretching moves
every boundary by the same factor, so one ASR pass serves every candidate
stretch - and a search is needed because truncation changes the mora-to-speech
ratio. Stretching to exactly 6.0 and then cutting overshot to 4.50, i.e.
drawling.
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

p = argparse.ArgumentParser()
p.add_argument("--source", default="ref_short.wav")
p.add_argument("--target-rate", type=float, default=6.0,
               help="moras per second of speech. ~6 is narration pace")
p.add_argument("--max-seconds", type=float, default=4.6,
               help="4.6 leaves 6.4s of chain headroom, over the 5.5s mean chunk")
p.add_argument("--name", default="ref_calm_short")
p.add_argument("--model", default="small")
p.add_argument("--show-asr", action="store_true")
a = p.parse_args()

REF = HERE / "ref"
src = REF / a.source
truth = src.with_suffix(".txt").read_text(encoding="utf-8").strip()
words = truth.split()

y, sr = sf.read(str(src), dtype="float32")
if y.ndim > 1:
    y = y.mean(axis=1)


def articulation(sig, text, thresh=-38.0):
    """Moras per second of ACTUAL SPEECH - silence excluded."""
    win = int(sr * 0.02)
    env = np.array([20 * np.log10(max(np.sqrt(np.mean(sig[i:i + win] ** 2)), 1e-9))
                    for i in range(0, len(sig) - win, win)])
    return pacing.moras(text) / max((env >= thresh).sum() * 0.02, 1e-6), \
        (env >= thresh).sum() * 0.02


rate0, sp0 = articulation(y, truth)
print(f"source      : {src.name}  {len(y)/sr:.2f}s, {sp0:.2f}s speech, "
      f"{rate0:.2f} moras/sec")

print(f"\nword timestamps (faster-whisper '{a.model}', CPU so the GPU stays free)")
from faster_whisper import WhisperModel
m = WhisperModel(a.model, device="cpu", compute_type="int8")
segs, _ = m.transcribe(str(src), language="mr", word_timestamps=True)
asr = [w for s_ in segs for w in (s_.words or [])]


def norm(w):
    return re.sub(r"[^ऀ-ॿ]", "", w)


tn, an = [norm(w) for w in words], [norm(w.word) for w in asr]
print(f"  ASR {len(asr)} words, transcript {len(words)}")
if a.show_asr:
    for w in asr:
        print(f"    {w.start:5.2f}-{w.end:5.2f}  {w.word}")

# Exact-match alignment is hopeless here: Whisper's Marathi renders वाजता as
# वास्ता and splits लॅपटॉपवर into two tokens, so only 4 of 11 words matched and
# the result was a useless two-word clip.
#
# But only a PREFIX is needed, and errors accumulate left to right - the first
# six tokens map 1:1 and the split happens at word seven. So map positionally
# and guard it with per-word SIMILARITY: a prefix is trustworthy as long as
# every pair up to the cut still resembles each other. Refuse beyond that
# rather than cut on a boundary the ASR has already drifted past.
SIM = 0.45
end_of, good_prefix = {}, 0
for i in range(min(len(words), len(asr))):
    r = difflib.SequenceMatcher(None, tn[i], an[i]).ratio()
    if r < SIM:
        break
    end_of[i] = asr[i].end
    good_prefix = i + 1
print(f"  positional alignment holds for the first {good_prefix}/{len(words)} words"
      f"  (similarity >= {SIM})")
if good_prefix < 2:
    sys.exit("  could not align ASR to your transcript - refusing to write a\n"
             "  clip whose text does not match its audio. Try --model medium,\n"
             "  or record a short calm clip directly.")

_DANGLING = {"ते", "तो", "ती", "हे", "हा", "ही", "आणि", "पण", "व", "का",
             "मग", "जे", "जो", "जी", "त्या", "या", "अन्", "मध्ये"}


def trial(stretch):
    """What would this stretch produce, after truncation at a word boundary?"""
    ok = [i for i in sorted(end_of) if end_of[i] / stretch <= a.max_seconds]
    if not ok:
        return None
    n = ok[-1] + 1
    while n > 1 and words[n - 1].strip(".,;:!?—–\"'") in _DANGLING:
        n -= 1                                   # never end on a bare pronoun
    if n < 2 or (n - 1) not in end_of:
        return None
    cut = end_of[n - 1] / stretch + 0.08         # a breath past the last word
    txt = " ".join(words[:n])
    if not txt.rstrip().endswith(("।", ".", "?", "!")):
        txt = txt.rstrip(",;:—–") + "."
    speech = articulation(y[:int(end_of[n - 1] * sr)], txt)[1] / stretch
    return dict(stretch=stretch, n=n, cut=cut, text=txt,
                rate=pacing.moras(txt) / max(speech, 1e-6))


best, seen = None, 0
for i in range(20, 81):
    t = trial(i / 100.0)
    if t and t["cut"] <= a.max_seconds:
        seen += 1
        if best is None or abs(t["rate"] - a.target_rate) < abs(best["rate"] - a.target_rate):
            best = t
if not best:
    sys.exit("no stretch produces a clip under --max-seconds with 2+ words")

print(f"\n  searched {seen} stretch factors; best hits {best['rate']:.2f} "
      f"moras/sec against a {a.target_rate:.2f} target")
print(f"  stretch {best['stretch']:.2f}, keeping {best['n']}/{len(words)} words, "
      f"cut at {best['cut']:.2f}s")

ys = librosa.effects.time_stretch(y, rate=best["stretch"], n_fft=2048, hop_length=512)
ys = (ys / (np.max(np.abs(ys)) or 1.0) * 0.95).astype(np.float32)
final = ys[:int(best["cut"] * sr)]

out = REF / f"{a.name}.wav"
sf.write(str(out), final.astype(np.float32), sr)
out.with_suffix(".txt").write_text(best["text"], encoding="utf-8")

r2, _ = articulation(final, best["text"])
dur = len(final) / sr
print(f"\nwrote {out.name}  {dur:.2f}s, {r2:.2f} moras/sec")
print(f"  text: {best['text']}")
print(f"  chain headroom {11.0 - dur:.1f}s"
      f"{'  - enough for a 5.5s chunk' if 11.0 - dur >= 5.5 else '  - STILL TIGHT'}")
