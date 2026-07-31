# -*- coding: utf-8 -*-
"""Build a targeted recording script for a second training pass.

The model mispronounces ळ (and to a lesser degree ण, च) because it barely saw
them: 192 ळ against 1272 ल in 1.26 hours of audio, and two thirds of the
ळ-words in the stories never appeared in training at all.

Recording six more hours of general narration would fix that eventually. This
is the cheap version: pull sentences OUT OF YOUR OWN STORY SCRIPTS that are
dense in the sounds the model is weak on, ranked so the most valuable come
first. Reading 20-30 minutes of this is worth far more than hours of ordinary
material.

Two extra benefits, both of which matter:

  * The transcript is exact. It is your own written text, so there is no
    Whisper pass and no transcription errors to inherit.
  * The sentences are real narration in your own voice and register, not
    drilled word lists, so the model learns the phoneme in the prosody you
    actually use.

    python tools/make_recording_script.py
    python tools/make_recording_script.py --minutes 30 --target ळ ण

Writes recording_script.txt - read it aloud, one numbered line at a time.
"""
import argparse
import glob
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")

p = argparse.ArgumentParser()
p.add_argument("--minutes", type=float, default=25.0,
               help="roughly how long the finished recording should be")
p.add_argument("--target", nargs="*", default=["ळ", "ण"],
               help="the characters the model is weak on")
p.add_argument("--stories", default=str(HERE / "queue" / "done" / "*.txt"))
p.add_argument("--train", default=str(HERE / "transcripts" / "*.txt"))
p.add_argument("--out", default=str(HERE / "recording_script.txt"))
a = p.parse_args()

story_text = ""
for f in glob.glob(a.stories):
    story_text += Path(f).read_text(encoding="utf-8") + "\n"
train_text = ""
for f in glob.glob(a.train):
    train_text += Path(f).read_text(encoding="utf-8") + "\n"

if not story_text.strip():
    sys.exit(f"No story scripts matched {a.stories}")

# candidate sentences: real lines from the stories, sane length to read in one
# breath, no stage directions
SD = re.compile(r"^\s*\*?\s*\[.*?\]\s*\*?\s*$")
sents = []
for line in story_text.split("\n"):
    line = line.strip()
    if not line or SD.match(line):
        continue
    for s in re.split(r"(?<=[।\.\?\!])\s+", line):
        s = s.strip().strip('"“”')
        if 25 <= len(s) <= 150 and re.search(r"[ऀ-ॿ]", s):
            sents.append(s)

seen_words = set(re.findall(r"[ऀ-ॿ]+", train_text))
story_words = Counter(re.findall(r"[ऀ-ॿ]+", story_text))

# how badly each target character is under-represented
print("coverage in the current training set:")
for ch in a.target:
    tr, st = train_text.count(ch), story_text.count(ch)
    trd = len(re.findall(r"[ऀ-ॿ]", train_text)) or 1
    std = len(re.findall(r"[ऀ-ॿ]", story_text)) or 1
    print(f"  {ch}   training {tr:5,} ({tr/trd*1000:5.2f}/1000)   "
          f"stories {st:6,} ({st/std*1000:5.2f}/1000)")


def score(s):
    """Value of recording this sentence.

    Target characters are worth a lot; a target-bearing WORD the model has
    never seen is worth much more, because that is the actual gap. Divided by
    length so we buy the most coverage per second of your voice.
    """
    v = 0.0
    for ch in a.target:
        v += s.count(ch) * 3.0
    for w in re.findall(r"[ऀ-ॿ]+", s):
        if any(ch in w for ch in a.target):
            v += 6.0 if w not in seen_words else 1.5
            v += min(story_words[w], 40) * 0.15      # words you actually use a lot
    return v / (len(s) ** 0.5)


# greedy pick, penalising words already chosen so the list keeps broadening
chosen, used = [], Counter()
pool = sorted(set(sents), key=score, reverse=True)
CHARS_PER_MIN = 730.0            # measured: ~12 Devanagari chars/sec of narration
budget = a.minutes * CHARS_PER_MIN
total = 0
while pool and total < budget:
    # -inf, not -1: once repeat penalties push every candidate negative we
    # still want the least-bad one, otherwise the script stops far short of
    # the requested length
    best, best_v = None, float("-inf")
    for s in pool[:400]:                       # only rescore the plausible head
        v = score(s)
        for w in re.findall(r"[ऀ-ॿ]+", s):
            if any(ch in w for ch in a.target):
                v -= used[w] * 2.0             # diminishing returns on repeats
        if v > best_v:
            best, best_v = s, v
    if best is None:
        break
    pool.remove(best)
    chosen.append(best)
    total += len(best)
    for w in re.findall(r"[ऀ-ॿ]+", best):
        if any(ch in w for ch in a.target):
            used[w] += 1

new_words = {w for w in used if w not in seen_words}
hits = sum(best.count(ch) for best in chosen for ch in a.target)

header = f"""# Recording script - targeted top-up for the weak sounds
#
# {len(chosen)} sentences, {total:,} characters, roughly {total/CHARS_PER_MIN:.0f} minutes to read.
# Covers {len(used)} distinct words containing {' '.join(a.target)},
# of which {len(new_words)} were NEVER in the original training data.
# {hits} occurrences of the target characters, against {sum(train_text.count(c) for c in a.target)} in the whole current training set.
#
# HOW TO RECORD
#   Same mic, same room, same distance as the original recordings - the model
#   is learning your voice AND your room, so a change in either is noise.
#   Read at the pace you want stories narrated. That is slower than you think:
#   your current reference clip is 8.3 moras/sec and it makes everything brisk.
#   Leave a clear second of silence between numbered lines so the clips can be
#   cut apart automatically.
#   One take per line is fine. If you fluff a line, pause and say it again -
#   keep the good one when cutting.
#
# The transcript is this file, exactly as written. Do not paraphrase; every
# word you change has to be corrected by hand afterwards.
"""

body = "\n".join(f"{i:03d}. {s}" for i, s in enumerate(chosen, 1))
Path(a.out).write_text(header + "\n" + body + "\n", encoding="utf-8")

print(f"\n{len(chosen)} sentences, ~{total/CHARS_PER_MIN:.0f} min to read")
print(f"  distinct target words covered : {len(used)}")
print(f"  of those, never seen before   : {len(new_words)}")
print(f"  target-character occurrences  : {hits} "
      f"(current training set has {sum(train_text.count(c) for c in a.target)})")
print(f"\nwritten: {a.out}")
