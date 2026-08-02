# Marathi Story Voice — Mac mini M4

Same app as the PC. Detects Apple Silicon and runs on the M4 GPU via **MPS**,
in float32.

---

## 1. Install (once)

Copy this whole folder to the Mac, open Terminal in it, and run:

```bash
bash setup_mac.sh
```

That creates `~/marathi_tts`, builds a Python environment, installs PyTorch +
F5-TTS + cmudict, and copies the app and reference clips into place.

## 2. Get the model files

**The model you want is `model_voice_merged50.pt`.** Copy it across from the PC
(`C:\marathi_tts_models\`) into `~/marathi_tts/models/`, along with `vocab.txt`.
That plus a reference clip is everything the app needs.

| File | Where |
|---|---|
| `model_voice_merged50.pt` | the PC's `marathi_tts_models\` — 2.7 GB |
| `vocab.txt` | same folder |

> ### Why a merged model, and not the fine-tune
> The straight fine-tune had the right voice and broken Marathi — `थांबलं` read
> as `थांबला`, `कोल्हापूर` with a `ळ`, `अनन्या` as "anya". IndicF5 already knew
> those words from hundreds of hours of training; 80 epochs over 1.26 hours of
> one voice **overwrote** that knowledge. Classic catastrophic forgetting: the
> timbre is learned, the language is lost.
>
> The fix is weight interpolation, not retraining:
> ```
> merged = 0.5 × fine-tune + 0.5 × base IndicF5
> ```
> Speaker identity lives in a small part of the weights; pronunciation is spread
> across all of them, so blending half-way back recovers the language and keeps
> the voice. Judged by ear at 0.5: pronunciation correct, voice ~90% intact.

**To rebuild it on the Mac instead of copying** — needs `model_last_slim.pt` and
a HuggingFace login with IndicF5 access:

```bash
cd ~/marathi_tts
venv/bin/hf auth login
venv/bin/python - <<'EOF'
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch, shutil
p = hf_hub_download("ai4bharat/IndicF5", "model.safetensors")
base = load_file(p)
# keep the ema_model. prefix, strip only the compile wrapper and the vocoder
out = {k.replace("_orig_mod.", ""): v for k, v in base.items()
       if not k.startswith("vocoder.")}
torch.save({"ema_model_state_dict": out}, "models/indicf5_base.pt")
EOF
venv/bin/python tools/merge_ckpt.py \
  --base models/indicf5_base.pt --tuned models/model_last_slim.pt \
  --alpha 0.5 --out-dir models
mv models/model_last_slim_merged50.pt models/model_voice_merged50.pt
```

`--sweep` writes several blends if you want to audition the trade-off yourself.
Higher alpha keeps more of your voice and more of the mispronunciation.

The raw Kaggle outputs (`model_last.pt`, `model_8000.pt`) are only needed if you
want to redo the merge from scratch.

> ### The `.zip` trap — read this
> Your browser will save the checkpoint as **`model_last.zip`**.
> **That file already IS the `.pt`.** A PyTorch checkpoint is a ZIP archive
> internally, so browsers mislabel it. **Do not unzip it** — if you do you get a
> folder of `data.pkl` and `data/0…N` fragments, which is useless.
>
> Just rename it:
> ```bash
> mv ~/Downloads/model_last.zip ~/marathi_tts/models/model_last.pt
> ```
> (This cost us a wasted 10 GB extraction on the PC. Skip that step.)

## 3. Run

```bash
~/marathi_tts/start_ui.command
```

or double-click `start_ui.command` in Finder, then open
<http://127.0.0.1:7860>.

macOS may block it the first time ("unidentified developer") —
**System Settings → Privacy & Security → Open Anyway**.

The review UI is a separate app on its own port — play a finished story, flag
what sounds wrong, and it re-renders just those chunks:

```bash
~/marathi_tts/venv/bin/python ~/marathi_tts/review.py    # http://127.0.0.1:7861
```

It only loads the model when you press Repair, so it is safe to browse and flag
while a batch render is running.

---

## 4. Updating later

Quit the app first (Ctrl-C in its Terminal window, or just close it), then:

```bash
cd ~/Marathi_indicaf5_ui && bash mac/update_mac.sh
```

That pulls the latest code and copies it into `~/marathi_tts`. It **does not
touch** `models/`, `ref/`, `out/`, `queue/`, `venv/`, or your
`pronunciation.json` — your voice, your clips, your overrides and your rendered
audio all stay exactly as they are.

If you want a specific release rather than the tip:

```bash
cd ~/Marathi_indicaf5_ui && git fetch --tags && git checkout v1.1.0 && bash mac/update_mac.sh
```

To go back to the tip afterwards: `git checkout main`.

Releases are listed in [CHANGELOG.md](../CHANGELOG.md). If an update ever sounds
worse than what you had, `git checkout v1.0.0 && bash mac/update_mac.sh` puts the
previous version back — the model is untouched, so the comparison is fair.

> If you cloned to a different folder, use that path instead of
> `~/Marathi_indicaf5_ui`. `git remote -v` inside a folder confirms you're in
> the right one.

---

## What's included here

```
app.py                the web UI
translit.py           English -> Devanagari transliteration
numerals.py           numbers -> Marathi words (304 -> तीनशे चार)
pacing.py             syllable-based duration budget (stops the rushing)
review.py             listen, flag bad chunks, repair them (port 7861)
tools/merge_ckpt.py   blend a fine-tune back toward its base model
tools/repair.py       re-render individual chunks and rebuild a story
run_queue.py          headless overnight batch runner
setup_mac.sh          installer
update_mac.sh         updater for an existing install
start_ui.command      double-click launcher
pronunciation.json    your saved pronunciation overrides (if any)
ref/                  reference clip transcripts (the .wav files are not in git)
```

Reference clips shipped:

| clip | length | why |
|---|---|---|
| `ref_short.wav` | 5.6 s | **use this one** — every chunk re-synthesises the reference, so a shorter clip is faster |
| `ref_9s.wav` | 9.0 s | longer alternative |

Each `.wav` has a matching `.txt` holding the **exact** words spoken. Keep the
pairs together — the app reads the transcript by filename, and its accuracy
affects quality more than almost any other setting.

---

## Things learned the hard way (don't undo these)

- **float32 is deliberate.** In float16 this model emits `NaN`, which becomes a
  flat DC line — a file that looks fine, has the right length, and plays as
  total silence. That bug cost hours. Do not "optimise" it back to half
  precision.
- **`PYTORCH_ENABLE_MPS_FALLBACK=1`** is set for you, so any op Apple hasn't
  implemented on MPS quietly runs on CPU instead of crashing.
- **Never store the model or venv on a FAT32 / removable drive.** On the PC the
  external drive both capped files at 4 GB and disconnected mid-run, which
  killed long generations. Keep everything on the Mac's internal SSD.

## Speed

The M4 should be clearly faster than the PC's GTX 1650 — which spends most of a
long run thermally throttled. Use `ref_short.wav`: every chunk re-synthesises the
reference, so a 5.6 s clip beats a 9 s one by a wide margin over 200 chunks.

**NFE 32 is the default and the right setting.** 16 is a draft; past ~40 the
solver over-sharpens and adds buzzy artifacts, so more is not better. Use
**Estimate time** before committing to a full story.

Every chunk is written to `~/marathi_tts/out/parts/<timestamp>/` as it
completes, so a long run can never lose all its work.
