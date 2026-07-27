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

Download these from your Kaggle notebook's **Output** tab into
`~/marathi_tts/models/`:

| File | Where it is on Kaggle |
|---|---|
| `model_last.pt` | `F5-TTS/ckpts/marathi_voice/model_last.pt` |
| `vocab.txt` | `base/vocab.txt` |

Optionally also `model_8000.pt` (the mid-training checkpoint) — with ~1 hour of
training data the final epoch can overfit, so it is genuinely worth comparing
the two.

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

---

## What's included here

```
app.py                the web UI
translit.py           English -> Devanagari transliteration
setup_mac.sh          installer
start_ui.command      double-click launcher
pronunciation.json    your saved pronunciation overrides (if any)
ref/                  reference clips + their transcripts
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

The M4 should be clearly faster than the PC's GTX 1650. Start with **NFE 16**
and `ref_short.wav`; raise NFE to 32 for a final render. Use **Estimate time**
before committing to a full story.

Every chunk is written to `~/marathi_tts/out/parts/<timestamp>/` as it
completes, so a long run can never lose all its work.
