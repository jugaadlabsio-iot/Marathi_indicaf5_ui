# -*- coding: utf-8 -*-
"""
A/B the paragraph fix on the passage that exposed it: the grandmother/grandson
exchange in 28_The_Munjya, five consecutive one-line paragraphs.

    python tools/ab_paragraph.py

A = old behaviour  - blank line means 800ms AND a prosody reset
B = new behaviour  - a short/dialogue paragraph is a breath, chain continues
"""
import importlib.util
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("mtts_app", HERE / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

# Paragraphs 6-12 of 28_The_Munjya, verbatim, blank lines intact - this is the
# exchange that reads as five unrelated statements.
PASSAGE = """आदित्य लहानपणी गावी यायचा. उन्हाळ्यात. आंबे खायला, विहिरीवर पोहायला, आजीच्या हातचं जेवायला.

आजी म्हणायची — "त्या झाडावर मुंज्या राहतो. त्याला त्रास देऊ नकोस."

"मुंज्या म्हणजे काय, आजी?"

"एक मुलगा. तुझ्याच वयाचा. खूप वर्षांपूर्वी. त्याचं मुंज झालं — पण लग्न व्हायच्या आधीच तो गेला."

"त्याचं नाव काय?"

"विनू. विनायक देशमुख. आपल्या शेजारच्या देशमुखांचा. गेल्याला शंभर वर्षं झाली."

आदित्यला लहानपणी हे गोष्टीसारखं वाटायचं. भुताच्या गोष्टी. आजीच्या."""

ckpt = next(c for c in app.list_ckpts() if "voice_merged" in c.lower())
import soundfile as sf
ref = min(app.list_refs(), key=lambda r: sf.info(r).duration)
ref_txt = app.ref_text_for(ref)
d = app.load_dict()

VARIANTS = [
    ("A_old_para_break", dict(chain_across_paragraphs=False),
     {"chunk": 250, "flow": 60, "sent": 300, "line": 350, "spara": 800, "para": 800, "end": 0}),
    ("B_new_dialogue_flow", dict(chain_across_paragraphs=True),
     {"chunk": 250, "flow": 60, "sent": 300, "line": 350, "spara": 320, "para": 600, "end": 0}),
]

for name, kw, pauses in VARIANTS:
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    out, dur, took, n, run_dir = app.synthesize(
        PASSAGE, ckpt, ref, ref_txt, 1.0, 32, 400, pauses,
        True, [[k, v] for k, v in d.items()], True, app.DEVICE,
        out_name=f"_ab_para_{name}",
        on_progress=lambda f, m: print(f"    {m}", end="\r", flush=True),
        log=lambda m: None,
        cfg=2.0, pace=1.35, seed=7, seed_per_chunk=False,
        max_secs=8.0, lead_ms=350, tail_ms=900,
        chain=True, chain_reanchor=8, apply_warmth=True, **kw)
    print(f"\n  {out.name}   {dur:.1f}s audio, {n} chunks, {time.time()-t0:.0f}s")
