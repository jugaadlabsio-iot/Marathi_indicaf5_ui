# -*- coding: utf-8 -*-
"""Acoustic post-processing - the gap between "clean" and "in the room".

Vocos decodes accurate speech that is also dry and thin: flat low end, a
slightly brittle top, and volume that jumps between a whispered line and a
shouted one. Commercial systems put a mastering chain after the vocoder;
this is that chain, kept deliberately gentle.

Three stages, in the order a mastering engineer would use them:

  1. WARMTH      a shelf around 150-250 Hz for chest resonance
  2. DE-HARSH    a soft roll-off above ~11 kHz, where diffusion artefacts and
                 sibilance live - not a brick wall, which sounds muffled
  3. COMPRESS    2:1 soft-knee, so a quiet line and a loud one sit closer
                 together without the pumping a hard limiter gives

Nothing here is destructive: it reads a wav and writes a new one.

    python tools/warmth.py --ab out/story.wav      # A/B pair to audition
    python tools/warmth.py out/story.wav           # process in place -> _warm.wav
    python tools/warmth.py --all                   # every wav in out/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

HERE = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")


def low_shelf(x, sr, f0=200.0, gain_db=2.5):
    """Lift the chest register. Shelf, not a peak - a bell here sounds boxy."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / 0.9 - 1) + 2)
    cw, sqA = np.cos(w0), np.sqrt(A)
    b = [A * ((A + 1) - (A - 1) * cw + 2 * sqA * alpha),
         2 * A * ((A - 1) - (A + 1) * cw),
         A * ((A + 1) - (A - 1) * cw - 2 * sqA * alpha)]
    a = [(A + 1) + (A - 1) * cw + 2 * sqA * alpha,
         -2 * ((A - 1) + (A + 1) * cw),
         (A + 1) + (A - 1) * cw - 2 * sqA * alpha]
    return signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], x)


def de_harsh(x, sr, cutoff=11000.0, order=2):
    """Gentle roll-off, not a wall. A steep low-pass reads as muffled; a
    2nd-order slope takes the edge off diffusion hiss and leaves air."""
    if cutoff >= sr / 2:
        return x
    sos = signal.butter(order, cutoff, "lowpass", fs=sr, output="sos")
    return signal.sosfilt(sos, x)


def compress(x, sr, thresh_db=-20.0, ratio=2.0, attack_ms=8.0,
             release_ms=140.0, knee_db=6.0, makeup=True):
    """2:1 soft-knee compressor with a level-detector envelope.

    Narration swings a long way between a whisper and a shout; a listener on
    headphones ends up riding the volume. Gentle compression removes that
    without the pumping a limiter introduces.
    """
    eps = 1e-9
    env = np.abs(x).astype(np.float64)
    a_att = np.exp(-1.0 / (sr * attack_ms / 1000.0))
    a_rel = np.exp(-1.0 / (sr * release_ms / 1000.0))

    # one-pole envelope follower, fast up and slow down
    y = np.empty_like(env)
    prev = 0.0
    for i in range(env.size):
        s = env[i]
        coef = a_att if s > prev else a_rel
        prev = coef * prev + (1 - coef) * s
        y[i] = prev

    lvl = 20 * np.log10(y + eps)
    over = lvl - thresh_db
    gain_db = np.zeros_like(lvl)

    # soft knee: ease into the ratio instead of switching at the threshold
    knee = over > -knee_db / 2
    hard = over > knee_db / 2
    within = knee & ~hard
    gain_db[within] = -((over[within] + knee_db / 2) ** 2) * (1 - 1 / ratio) / (2 * knee_db)
    gain_db[hard] = -over[hard] * (1 - 1 / ratio)

    out = x * (10 ** (gain_db / 20))
    if makeup:                      # put back what the ratio took off the top
        out *= 10 ** ((-thresh_db * (1 - 1 / ratio) * 0.5) / 20)
    return out


def process(x, sr, warmth_db=2.5, cutoff=11000.0, ratio=2.0, peak=0.95):
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    x = low_shelf(x, sr, gain_db=warmth_db)
    x = de_harsh(x, sr, cutoff=cutoff)
    x = compress(x, sr, ratio=ratio)
    m = float(np.max(np.abs(x)))
    if m > 0:
        x = x / m * peak
    return x.astype(np.float32)


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
