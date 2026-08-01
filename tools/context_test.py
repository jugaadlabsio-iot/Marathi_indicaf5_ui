# -*- coding: utf-8 -*-
"""Why is a word right on its own and wrong inside a story?

Every probe word came back correct in a short isolated phrase, including
कोल्हापूर exactly as spelled - yet both are wrong in the rendered stories. So
the failure is not in the word, the inflection or the spelling. It is in the
conditions the story render puts around it.

Three things differ between the two, and this separates them:

    A  the real story chunk, at the story's settings   (should be WRONG)
    B  the same chunk, at the probe's settings         (CFG/NFE?)
    C  the same words, alone in a short phrase         (chunk length?)

If B is right, the settings did it - CFG 2.2 or NFE 30.
If B is wrong and C is right, chunk length or position in the chunk did it.
If all three are wrong, the story text differs from the probe text in some way
we have not spotted yet, and the next step is to bisect the chunk.

    python tools/context_test.py
"""
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

out_dir = HERE / "out" / "context"
out_dir.mkdir(parents=True, exist_ok=True)
for old in out_dir.glob("*.wav"):
    old.unlink()

ckpt = str(app.MODELS_DIR / "model_last_slim.pt")
ref = min(app.list_refs(), key=lambda r: sf.info(r).duration)
ref_txt = app.ref_text_for(ref)
_, ref_norm, ref_sec = app.ref_profile(ref, ref_txt.strip())
rate = pacing.speech_rate(ref_norm, ref_sec)
tts = app.get_tts(ckpt, str(app.MODELS_DIR / "vocab.txt"), app.DEVICE)

# Real chunks from the stories, carrying the words reported wrong.
STORY_CHUNKS = [
    ("shala_full",  "शाळेच्या मागच्या बाजूला जुनी लायब्ररी होती. तिथे कोणी जात नव्हतं."),
    ("shala_short", "शाळेच्या मागे."),
    ("kolha_full",  "शुक्रवारी रात्री त्याला कोल्हापूरला जायचं होतं — आईचं बरं नव्हतं."),
    ("kolha_short", "तो कोल्हापूरला निघाला."),
]

SETTINGS = [
    ("A_story_cfg2.2_nfe30", dict(cfg_strength=2.2, nfe_step=30)),
    ("B_probe_cfg2.0_nfe32", dict(cfg_strength=2.0, nfe_step=32)),
    ("C_cfg1.8_nfe32",       dict(cfg_strength=1.8, nfe_step=32)),
]


def render(text, extra, pace=1.2):
    kw = dict(ref_file=ref, ref_text=ref_txt.strip(), speed=1.0,
              sway_sampling_coef=-1.0, remove_silence=False, seed=7,
              fix_duration=ref_sec + pacing.estimate_seconds(
                  text, rate, roominess=1.08 * pace))
    kw.update(extra)
    w, sr, _ = tts.infer(gen_text=text, **kw)
    w = np.asarray(w, dtype=np.float32)
    return app.fade_edges(app.trim_silence(w, sr), sr), sr


order = []
for sname, extra in SETTINGS:
    pieces, sr = [], 24000
    for cname, text in STORY_CHUNKS:
        w, sr = render(text, extra)
        print(f"  {sname:24} {cname:12} {len(w)/sr:4.1f}s  {text[:44]}")
        pieces.append(w)
        pieces.append(np.zeros(int(sr * 0.9), np.float32))
        order.append((sname, cname, text))
    audio = np.concatenate(pieces)
    audio = audio / (np.abs(audio).max() or 1.0) * 0.95
    sf.write(out_dir / f"{sname}.wav", audio, sr)
    print(f"  -> {out_dir / (sname + '.wav')}\n")

(out_dir / "order.txt").write_text(
    "\n".join(f"{a}\t{b}\t{c}" for a, b, c in order), encoding="utf-8")
print(f"order: {out_dir / 'order.txt'}")
print("\nEach file has the same 4 items: shala_full, shala_short,")
print("kolha_full, kolha_short. Compare the SAME item across files.")
