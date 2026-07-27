# -*- coding: utf-8 -*-
"""Report whether a wav is digitally silent, inaudibly quiet, or fine."""
import sys, numpy as np, soundfile as sf

for path in sys.argv[1:]:
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:
        print(f"{path}: UNREADABLE ({e})")
        continue
    x = x.mean(1) if x.shape[1] > 1 else x[:, 0]
    if len(x) == 0:
        print(f"{path}: EMPTY FILE (0 samples)")
        continue
    peak = float(np.abs(x).max())
    rms = float(np.sqrt((x ** 2).mean()))
    nz = int((np.abs(x) > 1e-6).sum())
    dbfs = 20 * np.log10(peak) if peak > 0 else -999
    print(f"\n{path}")
    print(f"  {len(x)/sr:.2f}s @ {sr}Hz | peak {peak:.6g} ({dbfs:.1f} dBFS) | rms {rms:.6g}")
    print(f"  non-silent samples: {nz}/{len(x)} ({100*nz/len(x):.2f}%)")
    if peak == 0:
        print("  -> DIGITAL SILENCE (all zeros): model or vocoder produced nothing.")
    elif dbfs < -50:
        print("  -> HAS STRUCTURE BUT INAUDIBLE: amplitude/normalization problem — recoverable.")
    elif dbfs < -20:
        print("  -> quiet but audible; normalize and listen.")
    else:
        print("  -> normal level. If it sounds wrong, it's content not amplitude.")
    # where is the energy? crude voiced-band check
    if peak > 0:
        n = min(len(x), sr * 5)
        spec = np.abs(np.fft.rfft(x[:n] * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, 1 / sr)
        band = (freqs > 80) & (freqs < 4000)
        frac = spec[band].sum() / (spec.sum() + 1e-12)
        print(f"  energy in 80-4000 Hz (speech band): {100*frac:.1f}%"
              + ("  <- looks like speech" if frac > 0.5 else "  <- NOT speech-like (noise/DC)"))
