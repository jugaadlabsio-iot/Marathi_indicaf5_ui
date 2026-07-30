# 🎙️ Marathi Story Voice

**A local, private text-to-speech studio for Marathi audio storytelling — powered by your own fine-tuned voice.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![F5-TTS](https://img.shields.io/badge/F5--TTS-1.1.22-6f42c1)](https://github.com/SWivid/F5-TTS)
[![IndicF5](https://img.shields.io/badge/base%20model-AI4Bharat%20IndicF5-ff6f00)](https://huggingface.co/ai4bharat/IndicF5)
[![Gradio](https://img.shields.io/badge/UI-Gradio-F97316?logo=gradio&logoColor=white)](https://www.gradio.app/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20(M--series)-0078D6)](#-hardware)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MPS-000000?logo=apple&logoColor=white)](https://developer.apple.com/metal/pytorch/)
[![Language](https://img.shields.io/badge/language-मराठी-138808)](#)
[![Runs](https://img.shields.io/badge/runs-100%25%20offline-success)](#-privacy)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

> **Topics:** `marathi` · `tts` · `text-to-speech` · `voice-cloning` · `voice-fine-tuning` · `indicf5`
> `f5-tts` · `ai4bharat` · `devanagari` · `audiobook` · `storytelling` · `horror-stories`
> `gradio` · `pytorch` · `cuda` · `apple-silicon` · `offline-first`

---

## What this is

A finished, self-contained studio for turning **written Marathi scripts into narrated audio in your own voice**. The voice comes from an [IndicF5](https://huggingface.co/ai4bharat/IndicF5) model fine-tuned on ~1 hour of personal narration. Everything — model, inference, UI — runs **locally**; no audio, text, or voice data leaves the machine.

Built for a specific job: producing long-form Marathi horror-story narration (भयकथा) at repeatable quality, in batches, overnight.

| | |
|---|---|
| **Base model** | `ai4bharat/IndicF5` (F5-TTS architecture, 11 Indian languages) |
| **Fine-tuned on** | ~1.08 h clean single-speaker Marathi narration · 341 clips |
| **Training** | Free Kaggle T4 · ~6 h · 80 epochs — notebook included |
| **Inference** | NVIDIA CUDA · Apple Silicon MPS · CPU |
| **Weights** | **not in this repo** — see [Getting the model](#-getting-the-model) |

---

## ✨ Features

- **Web UI on `localhost`** — paste a script, press Generate.
- **Overnight batch queue** — line up many stories, run unattended, collect in the morning. A failing story never stops the run.
- **Automatic English → Devanagari** — driven by the CMU pronouncing dictionary (126k words), so it follows how a word is *said*, not how it is spelled: `buyer → बायर`, `Scientific → सायंटिफिक`, `IT → आय टी`.
- **Pronunciation dictionary** — word-level overrides that always beat the automatic conversion. The practical handle for Marathi's dental vs palatal च/ज.
- **Stage directions skipped** — `[वातावरणनिर्मिती ...]` and `*[सूचना: ...]*` cues are stripped, not narrated.
- **Numbers read as Marathi words** — `३०४ → तीनशे चार`, `1994 → एकोणीसशे चौऱ्याण्णव`, `2019 → दोन हजार एकोणीस`, `MH 09 BT → एम एच शून्य नऊ बी टी`, `११:४५ → अकरा पंचेचाळीस`. Digits are barely present in the training data, so left alone they come out wrong or get skipped.
- **Syllable-based pacing** — speech time is budgeted from real Devanagari syllables rather than F5-TTS's UTF-8 byte count, which under-allots plain text by more than half. This is the fix for rushed delivery, sentences running together, clipped final words, and words going missing.
- **Single-pass chunking** — chunk size is capped to what F5-TTS can render in one forward pass, so it never splits and cross-fades behind your back. This is the fix for broken sentences and slurred words.
- **Layout-driven pacing** — your line breaks and blank lines become real pauses, each join fade-matched so it cannot click.
- **Reference clip manager** — record or upload clips per mood (calm, tense, dialogue) with their transcripts; switching clips is the main control over delivery.
- **Crash-resilient** — every chunk is written to disk as it finishes, so a long run can never lose all its work.
- **Delivery presets** — whisper 0.80 → chase 1.15, plus NFE and CFG controls.

---

## 🚀 Quick start

### Windows + NVIDIA

```bash
git clone https://github.com/jugaadlabsio-iot/Marathi_indicaf5_ui.git
cd Marathi_indicaf5_ui
python -m venv venv && venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Put `model_last.pt` and `vocab.txt` in `C:\marathi_tts_models\`, then:

```bash
start_ui.bat          # -> http://127.0.0.1:7860
```

### macOS (Apple Silicon)

```bash
bash mac/setup_mac.sh
```

Full walkthrough: [`mac/README_MAC.md`](mac/README_MAC.md).

### Overnight batch (no browser needed)

```bash
run_queue.bat --nfe 32
```

---

## 📦 Getting the model

The fine-tuned weights are **deliberately excluded** — they are ~1.4 GB of trained model (5 GB with optimizer state) and are personal voice data.

Produce your own with [`notebooks/IndicF5_finetune_Kaggle.ipynb`](notebooks/IndicF5_finetune_Kaggle.ipynb) — it runs end to end on a free Kaggle T4 in ~6 hours:

1. Upload ~1 hour of clean single-speaker audio as a Kaggle Dataset
2. Whisper segments and transcribes it into training clips
3. IndicF5 is converted to an F5-TTS checkpoint and fine-tuned
4. Download `model_last.pt` + `vocab.txt` into your models folder

> **The `.zip` trap:** browsers save the checkpoint as `model_last.zip`. **That file already *is* the `.pt`** — a PyTorch checkpoint is a ZIP container internally. Do **not** unzip it; just rename it to `.pt`.

Shrink a training checkpoint for faster loading:

```bash
python tools/slim_ckpt.py model_last.pt model_last_slim.pt   # 5.02 GB -> 2.51 GB
```

---

## 🎛️ Parameters

| Setting | Range | Notes |
|---|---|---|
| **Speed** | 0.80 – 1.15 | 0.80 whisper · 0.85 atmospheric · 1.00 narration · 1.10 tense · 1.15 chase |
| **NFE steps** | 8 – 64 | 16 draft · **32 final**. Past ~40 the solver over-sharpens and adds buzzy artifacts — more steps is *not* more quality here |
| **CFG strength** | 1.0 – 3.0 | 1.5 loose · **2.0 balanced** · 2.5 tight voice match · 3.0 can sound stiff |
| **Chunk size** | 150 – 700 chars | Auto-capped when *single-pass chunks* is on — see below |
| **Pauses** | ms | Within-line 150 · line break 350 · paragraph 800 |
| **Read numbers as words** | on | ३०४ → तीनशे चार · 1994 → एकोणीसशे चौऱ्याण्णव · 09 → शून्य नऊ · ११:४५ → अकरा पंचेचाळीस |
| **Single-pass chunks** | on | See below |
| **Budget duration by syllables** | on | See below — this is the fix for rushing and dropped words |
| **Roominess** | 0.85 – 1.35 | Extra time on top of the estimate. Raise if delivery is hurried, lower if lines drawl |
| **One sentence per chunk** | on | Each sentence gets its own start, end and pause instead of running together |
| `sway_sampling_coef` | −1 | Fixed |

### Syllable-based duration — the fix for rushing and dropped words

F5-TTS decides how long a line may take **before it generates anything**, from
its UTF-8 byte count:

```python
duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)
```

For Devanagari that is close to meaningless. Every codepoint costs 3 bytes
whether or not it takes time to say — a matra, an anusvara and the halant all
cost 3 bytes, and the halant makes a word *shorter* by deleting the inherent
vowel. `कमल` is 3.0 bytes per syllable; `स्वप्न` is 6.9. A plain line gets less
than half the time a conjunct-heavy one gets.

Starved lines rush, start fast, run sentences together, clip their last word,
and past a point **drop words entirely**. Auditing two real stories with
`tools/pacing_report.py` found 103 of 103 and 102 of 103 lines under-allotted;
`"तर?"` had been given 0.3 seconds.

`pacing.py` counts real syllables instead, weights long vowels, nasals and
conjuncts, adds time for punctuation, calibrates against your reference clip's
**measured** speaking rate, and passes F5-TTS an explicit `fix_duration`.

Audit your own scripts:

```bash
python tools/pacing_report.py queue/*.txt
```

> Your whole output inherits the reference clip's tempo. If the clip is brisk,
> every story is brisk — the app warns you above ~7.5 moras/sec. Either raise
> **Roominess** or record a calmer reference.

### Single-pass chunks — the fix for broken sentences

F5-TTS budgets roughly **22 seconds of audio per forward pass**, reference clip
included. Hand it more text than that and it silently splits the chunk and
cross-fades the halves itself — and those hidden seams are where words slur,
sentences sound cobbled together, and "studio quality" starts buzzing. The log
gives it away: `Generating audio in 2 batches`.

With this on, the chunk size is derived from your reference clip
(`(22 s − clip length) × its own chars-per-second`, minus 10 % headroom), so
every chunk renders in exactly one pass and **every join is one we control**,
with a real pause and an 8 ms fade. A 6-second clip yields ~160 characters; a
9-second clip only ~118 — one more reason to keep reference clips short.

**Punctuation still does its work** — `.` short pause · `,` tiny · `...` dramatic · `—` beat · `!` energy · `?` rising tone. A space inside a compound word can fix its pronunciation.

**Reference clips: 6–10 seconds.** Every chunk re-synthesises the reference, so a 6 s clip is materially faster than a 12 s one. The transcript must match the clip **word for word** — this affects quality more than almost any other setting.

---

## 🖥️ Hardware

| Device | Works | Notes |
|---|---|---|
| NVIDIA ≥ 4 GB VRAM | ✅ | float32 enforced (see gotchas) |
| Apple Silicon (M1–M4) | ✅ | MPS, with CPU fallback for unimplemented ops |
| CPU only | ✅ | Slow but correct |

Measured on a **GTX 1650 (4 GB)**: ~0.35× realtime at NFE 32. A 12-minute story ≈ 30–40 min at NFE 16. Apple M4 is considerably faster.

---

## ⚠️ Hard-won gotchas

Documented because each cost real debugging time:

1. **float16 produces silent audio.** In fp16 this model emits `NaN`, which libsndfile writes as a constant `-1.0` — a file with the right length and zero sound. **float32 is enforced everywhere; do not "optimise" it back.**
2. **A PyTorch `.pt` is a ZIP.** Browsers label it `.zip`. Renaming is correct; unzipping gives a useless folder of `data.pkl` fragments.
3. **Never put the model or venv on FAT32 / removable storage.** FAT32 caps files at 4 GB (checkpoints are 5 GB), and a drive that disconnects mid-run kills hours of generation.
4. **Checkpoints are ~5 GB, not 1.4 GB** — they carry optimizer + EMA state. Save sparingly during training or you will fill a 20 GB Kaggle disk. Use `tools/slim_ckpt.py` for inference.
5. **Whisper transcripts contain errors** (`गणपतराव देशमुख` → `गनपतराव देशमोक`). Correcting them is the single biggest quality lever for a retrain.
6. **Free Kaggle needs Internet explicitly enabled** (phone-verified), and toggling it **resets the GPU accelerator**.

---

## 🗂️ Layout

```
app.py               Gradio UI - generation, queue, references, pronunciation
translit.py          English -> Devanagari (CMU pronouncing dictionary)
run_queue.py         Headless overnight batch runner
start_ui.bat         Launch the UI (Windows)
run_queue.bat        Launch the batch runner (Windows)
mac/                 Apple Silicon installer, launcher and guide
notebooks/           Kaggle fine-tuning notebook
ref/                 Reference transcripts (audio excluded)
tools/               Reference cutting, silence analysis, checkpoint slimming,
                     transcription, GPU diagnostics
```

---

## 🔒 Privacy

Fully offline at inference: the model, your scripts, your voice and the generated audio stay on your machine. Model weights, source recordings, generated audio and datasets are all git-ignored — this repository contains **code and documentation only**.

---

## 🙏 Credits

- [**AI4Bharat IndicF5**](https://huggingface.co/ai4bharat/IndicF5) — base model for 11 Indian languages
- [**F5-TTS**](https://github.com/SWivid/F5-TTS) (SWivid) — architecture and training/inference stack
- [**Vocos**](https://github.com/gemelo-ai/vocos) — neural vocoder
- [**faster-whisper**](https://github.com/SYSTRAN/faster-whisper) — dataset transcription
- [**CMUdict**](https://github.com/cmusphinx/cmudict) — English pronunciations
- [**Gradio**](https://www.gradio.app/) — UI

---

## 📄 License

Code in this repository is licensed under the [Apache License 2.0](LICENSE).

Two things the licence does **not** cover:

- **The base model** carries [AI4Bharat's own licence](https://huggingface.co/ai4bharat/IndicF5) — check it before any commercial use.
- **The fine-tuned voice is not here, and is not covered.** A cloned voice is
  personal, biometric-adjacent data. Do not distribute the weights, and only
  ever fine-tune on a voice with that person's explicit consent.
