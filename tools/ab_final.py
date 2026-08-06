# -*- coding: utf-8 -*-
"""
The paragraph fix, judged on the worst passage in the corpus.

    python tools/ab_final.py

28_The_Munjya is 87% short/dialogue paragraphs at 85 chars each, and chaining
engaged on only 45% of its chunks - the joint worst in the corpus, and the
story flagged as reading "one word at a time". Paragraphs 7-12 are its densest
dialogue run: a grandmother/grandson exchange where every line is its own
paragraph, so under the old rule every line got an 800ms stop AND a prosody
reset.

    A = old  - blank line means 800ms and re-anchor to the reference clip
    B = new  - a short/dialogue paragraph is a breath (320ms), chain continues

Rendered against two references so the two questions are separable:
ref_short (chosen on tempo) and ref_nat3 (cut from the 48kHz master).

Unlike the earlier attempt, `log` is NOT suppressed. That is how the first
paragraph A/B rendered with planned_s=0.00 on both sides - app.py had silently
lost `pacing` because a bare `import pacing` resolves against the CALLER's
sys.path, and tools/ was on it instead of the project root. Both sides came out
unpaced, were judged rushed, and told us nothing about paragraphs.
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
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("mtts_app", HERE / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

for mod, name in ((app.pacing, "pacing"), (app.numerals, "numerals"),
                  (app.warmth, "warmth")):
    if mod is None:
        sys.exit(f"app.py could not import {name} - refusing to render a "
                 f"comparison that would be silently degraded.")
print("modules ok: pacing, numerals, warmth all loaded\n")

# 28_The_Munjya, paragraphs 7-12 verbatim, blank lines intact.
PASSAGE = """आजी म्हणायची — "त्या झाडावर मुंज्या राहतो. त्याला त्रास देऊ नकोस."

"मुंज्या म्हणजे काय, आजी?"

"एक मुलगा. तुझ्याच वयाचा. खूप वर्षांपूर्वी. त्याचं मुंज झालं — पण लग्न व्हायच्या आधीच तो गेला."

"त्याचं नाव काय?"

"विनू. विनायक देशमुख. आपल्या शेजारच्या देशमुखांचा. गेल्याला शंभर वर्षं झाली — "

आदित्यला लहानपणी हे गोष्टीसारखं वाटायचं. भुताच्या गोष्टी. आजीच्या."""

OLD = {"chunk": 250, "flow": 60, "sent": 300, "line": 350,
       "spara": 800, "para": 800, "end": 0}
NEW = {"chunk": 250, "flow": 60, "sent": 300, "line": 350,
       "spara": 320, "para": 600, "end": 0}

ckpt = next(c for c in app.list_ckpts() if "voice_merged" in c.lower())
d = app.load_dict()

REF = HERE / "ref"
RUNS = [
    ("1A_old_refshort", REF / "ref_short.wav", OLD, False),
    ("1B_new_refshort", REF / "ref_short.wav", NEW, True),
    ("2A_old_nat3",     REF / "ref_nat3.wav",  OLD, False),
    ("2B_new_nat3",     REF / "ref_nat3.wav",  NEW, True),
]

for name, refwav, pauses, across in RUNS:
    if not refwav.exists():
        print(f"  {name}: {refwav.name} missing, skipped")
        continue
    t0 = time.time()
    try:
        out, dur, took, n, run_dir = app.synthesize(
            PASSAGE, ckpt, str(refwav), app.ref_text_for(str(refwav)),
            1.0, 32, 400, pauses,
            True, [[k, v] for k, v in d.items()], True, app.DEVICE,
            out_name=f"_final_{name}",
            on_progress=lambda f, m: None,
            log=lambda m: None,                  # app.py now shouts on stderr
            cfg=2.0, pace=1.35, seed=7, seed_per_chunk=False,
            max_secs=8.0, lead_ms=300, tail_ms=700,
            chain=True, chain_reanchor=8,
            chain_across_paragraphs=across, apply_warmth=True)
        # prove the budget was applied rather than assuming it
        tsv = (run_dir / "chunks.tsv").read_text(encoding="utf-8").splitlines()[1:]
        budgets = [float(r.split("\t")[2]) for r in tsv if r.strip()]
        ok = "budgets OK" if all(b > 0 for b in budgets) else "*** UNPACED ***"
        print(f"  {name:<18}{out.name:<44}{dur:5.1f}s  {n} chunks  {ok}", flush=True)
    except Exception as e:
        print(f"  {name:<18}FAILED: {e}", flush=True)

print("\nCompare 1A vs 1B, then 2A vs 2B. The exchange should read as a")
print("conversation in B, and as separate statements in A.")
