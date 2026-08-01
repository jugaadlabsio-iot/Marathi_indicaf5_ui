# -*- coding: utf-8 -*-
"""Find out what this model can actually pronounce - by listening, not guessing.

Two experiments.

1. COMPARE CHECKPOINTS  (run this first)

       python tools/probe.py

   Renders the same problem words from every checkpoint in models/ and writes
   one wav per checkpoint. If model_8000 says ळ and model_last does not, the
   fine-tune overfit and you can simply use the earlier checkpoint - no
   retraining, no dataset work.

2. SEARCH FOR A SPELLING THAT WORKS

       python tools/probe.py --variants "शाळा=शाळा,शाल्हा,शाला" "फक्त=फक्त,फकत"

   Renders each spelling in turn. Whichever one sounds right becomes a
   pronunciation-dictionary entry: left column the real spelling, right column
   the one that reads correctly. The model reads characters, so respelling is
   a legitimate fix, not a hack.

Everything lands in out/probe/. Listen with the text in front of you.
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import importlib.util                                     # noqa: E402
import numpy as np                                        # noqa: E402
import soundfile as sf                                    # noqa: E402

spec = importlib.util.spec_from_file_location("mtts_app", HERE / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

# The sounds actually reported as wrong, each in a short carrier sentence so
# the phoneme is heard in context and not just in isolation.
# Inflected forms. The first probe tested base forms (शाळा, जवळ, वेळ) and every
# one came out right, yet story 2 was still wrong - and every ळ-word in story 2
# had been seen in training. What differs is the shape of the syllable: the
# stories use ळे, ळ्या, ळां, where the base forms use a bare ळ or ळा.
INFLECTED = [
    ("base  शाळा",    "शाळा बंद होती."),
    ("ळे    शाळेत",   "तो शाळेत गेला."),
    ("ळे    शाळेच्या", "शाळेच्या मागे विहीर होती."),
    ("ळे    शाळेतून",  "ती शाळेतून बाहेर पडली."),
    ("ळ्या  सगळ्यात",  "सगळ्यात शेवटी तो आला."),
    ("ळ्या  पावसाळ्यात", "पावसाळ्यात रस्ता बंद होतो."),
    ("ळा    मिळाल्या", "मला उत्तरं मिळाल्याशिवाय जाणार नाही."),
    ("ळां   वाघुळांचा", "वटवाघुळांचा आवाज आला."),
    ("ळ     गोळा",    "सगळे गोळा झाले."),
    ("ळ     मंगळवारी", "मंगळवारी शाळा बंद असते."),
    ("ळ     हळूहळू",  "हळूहळू अंधार पडत गेला."),
    ("ळ     कळत",     "मला काही कळत नव्हतं."),
    ("ळ     ओळखीची",  "ती ओळखीची वाटली."),
    ("ळ     वळला",    "तो मागे वळला."),
    ("ळ     पळाला",   "तो पळाला."),
    ("ल     वळला/वलला", "तो वळला, तो वलला."),
]

PROBES = [
    ("ळ  शाळा",   "शाळा बंद होती."),
    ("ळ  बाळ",    "बाळ रडत होतं."),
    ("ळ  जवळ",    "तो माझ्या जवळ आला."),
    ("ळ  वेळ",    "आता वेळ नाही."),
    ("ळ  डोळे",   "त्याचे डोळे लाल होते."),
    ("ळ  सकाळी",  "सकाळी सहा वाजता."),
    ("ल  vs ळ",   "काळ आणि काल वेगळे आहेत."),
    ("मी",        "मी तिथे गेलो होतो."),
    ("मी २",      "मी म्हणालो, मी एकटाच आहे."),
    ("च",         "चल, चार वाजले आहेत."),
    ("च vs क",    "चंद्र आणि कन्या."),
    ("फक्त",      "फक्त एकच माणूस होता."),
    ("जा",        "जा, आणि परत येऊ नकोस."),
    ("डॉक्टर",    "डॉक्टर आत आले."),
    ("ण",         "कोणी नव्हतं तिथे."),
    ("टक्के",     "पन्नास टक्के लोक गेले."),
]


def render(tts, ref, ref_txt, text, ref_sec, rate, pace):
    kw = dict(ref_file=ref, ref_text=ref_txt.strip(), speed=1.0,
              nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0,
              remove_silence=False, seed=7)
    import pacing
    kw["fix_duration"] = ref_sec + pacing.estimate_seconds(
        text, rate, roominess=1.08 * pace)
    wav, sr, _ = tts.infer(gen_text=text, **kw)
    wav = np.asarray(wav, dtype=np.float32)
    return app.fade_edges(app.trim_silence(wav, sr), sr), sr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="*", default=None,
                   help='e.g. "शाळा=शाळा,शाल्हा,शाला"')
    p.add_argument("--inflected", action="store_true",
                   help="probe the inflected ळ forms the stories actually use")
    p.add_argument("--ckpt", default=None, help="one checkpoint instead of all")
    p.add_argument("--pace", type=float, default=1.2)
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    out_dir = HERE / "out" / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = [a.ckpt] if a.ckpt else app.list_ckpts()
    if not ckpts:
        sys.exit(f"No checkpoints in {app.MODELS_DIR}")
    refs = app.list_refs()
    if not refs:
        sys.exit(f"No reference clip in {app.REF_DIR}")
    ref = min(refs, key=lambda r: sf.info(r).duration)
    ref_txt = app.ref_text_for(ref)
    _, ref_norm, ref_sec = app.ref_profile(ref, ref_txt.strip())
    import pacing
    rate = pacing.speech_rate(ref_norm, ref_sec)
    device = app.DEVICE if a.device == "auto" else a.device

    if a.variants:
        items = []
        for spec_ in a.variants:
            word, _, alts = spec_.partition("=")
            for alt in alts.split(","):
                alt = alt.strip()
                if alt:
                    items.append((f"{word} -> {alt}", alt))
        tag = "variants"
    elif a.inflected:
        items, tag = INFLECTED, "inflected"
    else:
        items, tag = PROBES, "probe"

    print(f"reference : {Path(ref).name}  {ref_sec:.2f}s  {rate:.2f} moras/sec")
    print(f"checkpoints: {len(ckpts)}   items: {len(items)}\n")

    index = []
    for ck in ckpts:
        name = Path(ck).stem
        print("=" * 60)
        print(f"{name}")
        print("=" * 60)
        tts = app.get_tts(ck, str(app.MODELS_DIR / "vocab.txt"), device)
        pieces, sr = [], 24000
        for label, text in items:
            t0 = time.time()
            wav, sr = render(tts, ref, ref_txt, text, ref_sec, rate, a.pace)
            print(f"  {label:14} {len(wav)/sr:4.1f}s  ({time.time()-t0:.0f}s)  {text}")
            pieces.append(wav)
            pieces.append(np.zeros(int(sr * 0.9), np.float32))   # gap to listen in
            index.append((name, label, text))
        audio = np.concatenate(pieces)
        peak = float(np.abs(audio).max()) or 1.0
        path = out_dir / f"{tag}__{name}.wav"
        sf.write(path, audio / peak * 0.95, sr)
        print(f"  -> {path}\n")

    listing = out_dir / f"{tag}__order.txt"
    listing.write_text(
        "\n".join(f"{n}\t{lab}\t{txt}" for n, lab, txt in index),
        encoding="utf-8")
    print(f"Order of items: {listing}")
    print("\nListen to each file with that list in front of you. For the")
    print("checkpoint comparison, the only question is: does one checkpoint")
    print("say ळ, मी and च correctly where the other does not?")


if __name__ == "__main__":
    main()
