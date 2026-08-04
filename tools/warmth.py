# -*- coding: utf-8 -*-
"""Apply the acoustic chain to files that are already rendered.

The processing itself lives in warmth.py at the project root, because the
render pipeline uses it too. This is just the command line around it.

    python tools/warmth.py --ab out/story.wav      # A/B pair to audition
    python tools/warmth.py out/story.wav           # -> story_warm.wav
    python tools/warmth.py --all                   # every wav in out/
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np                                        # noqa: E402
import soundfile as sf                                    # noqa: E402
from warmth import process                                # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wav", nargs="?", help="file to process")
    p.add_argument("--ab", metavar="WAV", help="write an A/B pair to audition")
    p.add_argument("--all", action="store_true", help="every wav in out/")
    p.add_argument("--warmth-db", type=float, default=2.5)
    p.add_argument("--cutoff", type=float, default=11000.0)
    p.add_argument("--ratio", type=float, default=2.0)
    p.add_argument("--seconds", type=float, default=45.0,
                   help="length of each A/B excerpt")
    a = p.parse_args()

    if a.ab:
        src = Path(a.ab)
        x, sr = sf.read(str(src))
        if x.ndim > 1:
            x = x.mean(axis=1)
        n = int(min(len(x), a.seconds * sr))
        # take from a minute in - openings are atypically quiet
        start = min(int(60 * sr), max(0, len(x) - n))
        clip = np.asarray(x[start:start + n], dtype=np.float32)
        out = HERE / "out" / "warmth"
        out.mkdir(parents=True, exist_ok=True)
        sf.write(out / "A_original.wav", clip / (np.abs(clip).max() or 1) * 0.95, sr)
        sf.write(out / "B_processed.wav",
                 process(clip, sr, a.warmth_db, a.cutoff, a.ratio), sr)
        print(f"  A_original.wav  / B_processed.wav   ({n/sr:.0f}s each)")
        print(f"  from {src.name}, starting at {start/sr:.0f}s")
        print(f"  settings: +{a.warmth_db} dB low shelf, {a.cutoff:.0f} Hz "
              f"roll-off, {a.ratio}:1 compression")
        print(f"  -> {out}")
        return

    targets = []
    if a.all:
        targets = sorted((HERE / "out").glob("*.wav"))
    elif a.wav:
        targets = [Path(a.wav)]
    if not targets:
        sys.exit("give a wav, or --ab <wav>, or --all")
    for t in targets:
        if t.stem.endswith("_warm"):
            continue
        x, sr = sf.read(str(t))
        y = process(x, sr, a.warmth_db, a.cutoff, a.ratio)
        dst = t.with_name(t.stem + "_warm.wav")
        sf.write(str(dst), y, sr)
        print(f"  {t.name} -> {dst.name}")


if __name__ == "__main__":
    main()
