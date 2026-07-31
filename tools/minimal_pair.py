# -*- coding: utf-8 -*-
"""Has the model actually lost the ळ / ल distinction?

Listening tells you a word sounds wrong. It does not tell you whether the
model CAN make the sound at all - and that is the difference between a fix
you can do with the pronunciation dictionary today and one that needs new
recordings.

The test is a controlled comparison. For a minimal pair like काळ / काल:

    within  = distance between two renders of THE SAME word on different seeds
              (this is the model's own run-to-run noise floor)
    between = distance between the two DIFFERENT words

If between is no bigger than within, the model is producing the same sound for
both spellings: the distinction is gone, and no respelling recovers it.
If between is clearly larger, the phoneme is in there and the failures are
word-specific - which the dictionary can fix.

    python tools/minimal_pair.py
    python tools/minimal_pair.py --ckpt C:\\marathi_tts_models\\model_8000.pt
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")

import importlib.util                                    # noqa: E402
import numpy as np                                       # noqa: E402
import soundfile as sf                                   # noqa: E402
import librosa                                           # noqa: E402

spec = importlib.util.spec_from_file_location("mtts_app", HERE / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
import pacing                                            # noqa: E402

PAIRS = [
    ("काळ", "काल", "ळ vs ल"),
    ("वेळ", "वेल", "ळ vs ल"),
    ("मूळ", "मूल", "ळ vs ल"),
    ("कोण", "कोन", "ण vs न"),
    ("चंद्र", "कंद्र", "च vs क"),
]
SEEDS = (11, 29)


def mel(wav, sr):
    m = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=64,
                                       n_fft=1024, hop_length=256)
    return librosa.power_to_db(m + 1e-10)


def distance(a, b):
    """Cosine distance between two mel spectrograms, length-normalised.

    Words differ in length, so both are resampled to a common frame count -
    crude next to DTW, but these are single short words and it is the same
    treatment for the within and between comparisons, which is what matters.
    """
    n = min(a.shape[1], b.shape[1])
    if n < 4:
        return float("nan")
    ai = np.stack([np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x)
                   for x in a])
    bi = np.stack([np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x)
                   for x in b])
    av, bv = ai.ravel(), bi.ravel()
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    return 1.0 - float(av @ bv / denom) if denom else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    ckpts = [a.ckpt] if a.ckpt else app.list_ckpts()
    ref = min(app.list_refs(), key=lambda r: sf.info(r).duration)
    ref_txt = app.ref_text_for(ref)
    _, ref_norm, ref_sec = app.ref_profile(ref, ref_txt.strip())
    rate = pacing.speech_rate(ref_norm, ref_sec)
    device = app.DEVICE if a.device == "auto" else a.device

    def say(tts, word, seed):
        text = word + "."
        kw = dict(ref_file=ref, ref_text=ref_txt.strip(), speed=1.0,
                  nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0,
                  remove_silence=False, seed=seed,
                  fix_duration=ref_sec + pacing.estimate_seconds(
                      text, rate, roominess=1.3))
        w, sr, _ = tts.infer(gen_text=text, **kw)
        w = np.asarray(w, dtype=np.float32)
        return app.trim_silence(w, sr), sr

    for ck in ckpts:
        name = Path(ck).stem
        print("=" * 68)
        print(name)
        print("=" * 68)
        tts = app.get_tts(ck, str(app.MODELS_DIR / "vocab.txt"), device)
        print(f"  {'pair':18} {'within':>8} {'between':>8}  {'ratio':>6}  verdict")
        for w1, w2, what in PAIRS:
            a1, sr = say(tts, w1, SEEDS[0])
            a2, _ = say(tts, w1, SEEDS[1])
            b1, _ = say(tts, w2, SEEDS[0])
            within = distance(mel(a1, sr), mel(a2, sr))
            between = distance(mel(a1, sr), mel(b1, sr))
            ratio = between / within if within else float("nan")
            if ratio < 1.15:
                v = "COLLAPSED - same sound for both spellings"
            elif ratio < 1.6:
                v = "weak - partly distinguished"
            else:
                v = "distinct - the model does make both sounds"
            print(f"  {w1}/{w2:8} {what:10} {within:8.4f} {between:8.4f} "
                  f"{ratio:6.2f}  {v}")
        print()

    print("within  = same word, two seeds  (the model's own noise floor)")
    print("between = the two different spellings")
    print("If between is no larger than within, the distinction is gone and no")
    print("respelling brings it back - that case needs new recordings.")


if __name__ == "__main__":
    main()
