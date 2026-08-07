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
from scipy.signal import butter, fftconvolve, filtfilt
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



def room(x, sr, wet=0.08, size_ms=55.0, damp_hz=4500.0, seed=7):
    """A microscopic room, added to speech that was decoded in a vacuum.

    Vocos reconstructs from a mel spectrogram and adds no space of its own, so
    what comes back carries only whatever room the reference clip had. Against
    a real recording it reads as slightly "in your head" rather than in a
    place.

    Deliberately tiny: early reflections only, no tail. A 55ms impulse is
    shorter than a syllable, so it thickens the voice without smearing
    consonants or sounding like a hall. Anything longer starts eating the
    retroflexes this project spent weeks getting right.

    0.08 chosen by ear against 0.0 and 0.15 on a real passage. The reference
    clip already carries some room of its own, so this is topping up a space
    that half exists rather than creating one - which is why the useful range
    is this narrow and why 0.15 was already too much.
    """
    if wet <= 0:
        return x
    rng = np.random.default_rng(seed)
    n = max(8, int(sr * size_ms / 1000.0))
    t = np.arange(n) / sr
    # a handful of decaying early reflections, not a dense tail
    ir = np.zeros(n, dtype=np.float64)
    for tap_ms, g in ((7.0, 0.55), (13.0, -0.38), (23.0, 0.27),
                      (31.0, -0.19), (43.0, 0.12)):
        i = int(sr * tap_ms / 1000.0)
        if i < n:
            ir[i] += g
    ir += rng.normal(0, 0.03, n) * np.exp(-t * 60.0)     # a little diffusion
    ir *= np.exp(-t * 45.0)
    # damp the reflections so they do not add sibilance back
    b, a_ = butter(2, min(damp_hz / (sr / 2.0), 0.99), btype="low")
    ir = filtfilt(b, a_, ir)
    ir /= max(np.sum(np.abs(ir)), 1e-9)

    wetsig = fftconvolve(x, ir, mode="full")[:x.size]
    return (1.0 - wet) * x + wet * wetsig


def process(x, sr, warmth_db=2.5, cutoff=11000.0, ratio=2.0, peak=0.95,
            room_wet=0.08):
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    x = low_shelf(x, sr, gain_db=warmth_db)
    x = de_harsh(x, sr, cutoff=cutoff)
    x = compress(x, sr, ratio=ratio)
    if room_wet > 0:
        x = room(x, sr, wet=room_wet)
    m = float(np.max(np.abs(x)))
    if m > 0:
        x = x / m * peak
    return x.astype(np.float32)
