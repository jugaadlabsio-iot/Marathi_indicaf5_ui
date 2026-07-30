# -*- coding: utf-8 -*-
"""Which lines of a script were being starved of time?

    python tools/pacing_report.py "C:\\path\\to\\story.txt"
    python tools/pacing_report.py queue/*.txt

For every sentence it prints what F5-TTS's byte-count estimate would have
allowed versus what the syllables actually need. Anything under 100% was being
rushed; badly starved lines are where words got dropped, sentences ran
together, and final consonants were clipped.

Read this against a story you already generated and disliked - the flagged
lines should be the ones that sounded wrong.
"""
import glob
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import soundfile as sf                                  # noqa: E402
import importlib.util                                   # noqa: E402

spec = importlib.util.spec_from_file_location("mtts_app", HERE / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
import pacing                                           # noqa: E402

paths = []
for pat in (sys.argv[1:] or [str(HERE / "queue" / "*.txt")]):
    paths.extend(glob.glob(pat))
if not paths:
    sys.exit("No files matched. Pass a script path.")

refs = app.list_refs()
if not refs:
    sys.exit(f"No reference clip in {app.REF_DIR}")
ref = min(refs, key=lambda r: sf.info(r).duration)
ref_txt = app.ref_text_for(ref)
_, ref_norm, ref_sec = app.ref_profile(ref, ref_txt.strip())
rate = pacing.speech_rate(ref_norm, ref_sec)
ref_bytes = len(ref_norm.encode("utf-8"))

print(f"reference : {Path(ref).name}  {ref_sec:.2f}s  {rate:.2f} moras/sec")
print(f"byte rate : {ref_bytes / ref_sec:.1f} bytes/sec "
      f"(what F5-TTS times by)\n")

worst_all = []
for p in paths:
    text = app.prepare_text(Path(p).read_text(encoding="utf-8"),
                            app.load_dict(), True, True)
    items = app.split_blocks(text, 400)
    print("=" * 72)
    print(Path(p).name)
    print("=" * 72)
    starved = []
    for chunk, _kind in items:
        need = pacing.estimate_seconds(chunk, rate)
        got = ref_sec / ref_bytes * len(chunk.encode("utf-8"))
        pct = got / need * 100 if need else 100
        if pct < 100:
            starved.append((pct, chunk, got, need))
    starved.sort()
    if not starved:
        print("  nothing starved - this script was already getting enough time\n")
        continue
    print(f"  {len(starved)} of {len(items)} lines under-allotted. Worst first:\n")
    for pct, chunk, got, need in starved[:12]:
        flag = "!!" if pct < 70 else ("! " if pct < 85 else "  ")
        print(f"  {flag} {pct:5.0f}%  had {got:4.1f}s, needs {need:4.1f}s   {chunk[:58]}")
    print()
    worst_all.extend(starved)

if worst_all:
    worst_all.sort()
    n_bad = sum(1 for p, *_ in worst_all if p < 70)
    print("=" * 72)
    print(f"{len(worst_all)} starved lines across {len(paths)} file(s); "
          f"{n_bad} severely (<70%).")
    print("Severely starved lines are where words go missing entirely.")
    print("Fix: keep 'Budget duration by syllables' on, and raise Roominess")
    print("if any still sound hurried.")
