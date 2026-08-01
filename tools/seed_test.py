# -*- coding: utf-8 -*-
"""Is the pronunciation a lottery?

Every controlled test so far has come back correct: base forms, inflected
forms, कोल्हापूर as-is, long chunks, short chunks, CFG 2.0 and 2.2, NFE 30 and
32. Yet the same words are wrong inside rendered stories.

One thing every one of those tests shared and the story run did not: a fixed
seed. probe.py and context_test.py pin seed=7. run_queue.py leaves seed None,
so F5TTS.infer draws a fresh random seed for every chunk:

    if seed is None:
        seed = random.randint(0, sys.maxsize)

Flow matching starts from noise. A different seed is a different starting
point, and on a model fine-tuned on 1.26 hours some starting points land on a
worse pronunciation than others. If that is what is happening, the same text
rendered on N seeds will be right on most and wrong on a few - and the fix is
not a dictionary or more data, it is to stop rolling the dice.

    python tools/seed_test.py
    python tools/seed_test.py --text "शाळेच्या मागे." --seeds 12
"""
import argparse
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

p = argparse.ArgumentParser()
p.add_argument("--text", default="शाळेच्या मागच्या बाजूला जुनी लायब्ररी होती.")
p.add_argument("--seeds", type=int, default=10)
p.add_argument("--cfg", type=float, default=2.0)
p.add_argument("--nfe", type=int, default=32)
p.add_argument("--pace", type=float, default=1.2)
a = p.parse_args()

out_dir = HERE / "out" / "seeds"
out_dir.mkdir(parents=True, exist_ok=True)
for old in out_dir.glob("*.wav"):
    old.unlink()

ckpt = str(app.MODELS_DIR / "model_last_slim.pt")
ref = min(app.list_refs(), key=lambda r: sf.info(r).duration)
ref_txt = app.ref_text_for(ref)
_, ref_norm, ref_sec = app.ref_profile(ref, ref_txt.strip())
rate = pacing.speech_rate(ref_norm, ref_sec)
tts = app.get_tts(ckpt, str(app.MODELS_DIR / "vocab.txt"), app.DEVICE)

fix = ref_sec + pacing.estimate_seconds(a.text, rate, roominess=1.08 * a.pace)
print(f"text : {a.text}")
print(f"cfg {a.cfg}  nfe {a.nfe}  {a.seeds} seeds\n")

# seed 7 first as the known-good control, then the kind of arbitrary values
# random.randint produces in a real run
seeds = [7] + [int(x) for x in
                np.random.default_rng(0).integers(1, 2**31 - 1, a.seeds - 1)]

pieces, sr, order = [], 24000, []
for k, sd in enumerate(seeds, 1):
    w, sr, _ = tts.infer(gen_text=a.text, ref_file=ref, ref_text=ref_txt.strip(),
                         speed=1.0, nfe_step=a.nfe, cfg_strength=a.cfg,
                         sway_sampling_coef=-1.0, remove_silence=False,
                         seed=sd, fix_duration=fix)
    w = np.asarray(w, dtype=np.float32)
    w = app.fade_edges(app.trim_silence(w, sr), sr)
    tag = " (control, known good)" if sd == 7 else ""
    print(f"  {k:2d}. seed {sd:<12} {len(w)/sr:4.1f}s{tag}")
    order.append(f"{k}\tseed {sd}{tag}")
    pieces.append(w)
    pieces.append(np.zeros(int(sr * 0.9), np.float32))

audio = np.concatenate(pieces)
audio = audio / (np.abs(audio).max() or 1.0) * 0.95
path = out_dir / "seeds.wav"
sf.write(path, audio, sr)
(out_dir / "order.txt").write_text("\n".join(order), encoding="utf-8")
print(f"\n-> {path}")
print("\nSame text every time; only the starting noise differs. Count how many")
print("of the takes get the word wrong. If it is some but not all, the story")
print("failures are seed luck, not a property of the word.")
