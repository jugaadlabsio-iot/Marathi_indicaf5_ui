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

Applied automatically at the end of every render (see synthesize), and
available on already-finished files through tools/warmth.py.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal



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
