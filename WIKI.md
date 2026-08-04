# Engineering wiki — Marathi Story Voice

Everything learned building this, in the order it was learned, with the numbers.
Written as context for whoever picks this up next, including future-me.

The short version: **most of the time went into problems that were not where
they appeared to be.** Silent audio was a dtype bug. Rushed narration was a
byte-count. Mispronunciation was catastrophic forgetting. Narration that sounded
like 200 separate recordings was the pipeline resetting a continuation mechanism
200 times. Each looked like something else first.

---

## Contents

1. [What this is](#1-what-this-is)
2. [Training on Kaggle](#2-training-on-kaggle)
3. [The silent audio bug](#3-the-silent-audio-bug)
4. [Text handling](#4-text-handling)
5. [The pacing crisis](#5-the-pacing-crisis)
6. [Chunking](#6-chunking)
7. [Edges and clipping](#7-edges-and-clipping)
8. [Seed variance](#8-seed-variance)
9. [Catastrophic forgetting — the big one](#9-catastrophic-forgetting--the-big-one)
10. [Prosody chaining](#10-prosody-chaining)
11. [Acoustic post-processing](#11-acoustic-post-processing)
12. [Gradio traps](#12-gradio-traps)
13. [Hardware](#13-hardware)
14. [Tools](#14-tools)
15. [Settings that matter](#15-settings-that-matter)
16. [Diagnostic method](#16-diagnostic-method)
17. [Still open](#17-still-open)

---

## 1. What this is

A local text-to-speech studio that reads Marathi horror stories in one specific
person's voice.

| | |
|---|---|
| base model | **IndicF5** (AI4Bharat) — F5-TTS architecture, 11 Indian languages, 2545-char vocab, gated on HuggingFace |
| framework | `f5-tts` 1.1.22, arch `F5TTS_Base` (dim 1024, depth 22, heads 16) |
| fine-tune data | **1.26 hours** of clean single-speaker audio, ~340 clips after VAD |
| trained on | Kaggle free T4, 80 epochs, loss 0.62 → 0.578, ~5.5 h |
| runs on | GTX 1650 4 GB (CUDA, float32) · Apple Silicon (MPS) · CPU |

The pipeline: script → text normalisation → chunking → per-chunk synthesis →
concatenation with measured pauses → one wav per story.

**F5-TTS synthesises each chunk as a continuation of whatever audio it is
conditioned on.** Given the same fixed reference clip every time - which is what
this pipeline did for a long while - each chunk restarts the prosody, and a
story reads as a list of separately-recorded sentences. §10 is how that was
fixed; it explains more of the design below than anything except §5.

---

## 2. Training on Kaggle

Ten blockers, in the order they bite. Full detail in `notebooks/`.

1. **Session options → Internet ON *and* GPU.** Toggling internet *resets the
   accelerator*. Recheck both. Without internet, pip failures look like
   "package not found" but are DNS.
2. **`%pip` not `!pip`** — `!pip` installs into a different environment than the
   kernel uses.
3. **Locate the dataset by glob**, it mounts unpredictably:
   `glob.glob("/kaggle/input/**/wav_clean", recursive=True)`
4. **Zip with Python `zipfile`,** not PowerShell `Compress-Archive` — the latter
   writes backslash entry names that Kaggle rejects.
5. **`prepare_csv_wavs.py` wants the metadata.csv *path*** (not its directory)
   and **absolute** audio paths.
6. **The safetensors → F5-TTS conversion** took four attempts. See §9 for the
   key layout; additionally you must add `ema["initted"]=tensor(True)` and
   `ema["step"]=tensor(0)`, also save `model_state_dict` with bare keys, and
   **not** leave top-level `step`/`update` keys — those make the trainer take
   the resume branch and demand an optimizer state that isn't there.
7. **`rm -rf ckpts/<dataset>` first** — the finetune CLI won't overwrite a stale
   `pretrained_*.pt`.
8. **Free Whisper's GPU memory before training**
   (`del model; gc.collect(); torch.cuda.empty_cache()`). It holds 3.8 GB and
   caused OOM at BATCH_FRAMES 3200; 2400 is safe once freed.
9. **Checkpoints are ~5.4 GB, not the 1.4 GB the model size suggests** —
   model + EMA + optimizer. Kaggle's ~20 GB working disk fills around epoch 16.
   Use `save_per_updates 8000`, `last_per_updates 1000`,
   `keep_last_n_checkpoints 1`.
10. **A `.pt` file *is* a zip.** Browsers save it as `model_last.zip`. Rename it,
    don't extract it. Extracting cost 10 GB and produced useless fragments.

**Data quality note.** Whisper large-v3 (mr) generated the transcripts and
misspells names (`गनपतराव देशमोक` for `गणपतराव देशमुख`). This was suspected as the
root of the pronunciation problems and **measured to be innocent** — see §9.

---

## 3. The silent audio bug

**Symptom:** generation completed, files had the right length, played as total
silence.

**Cause:** the model emits `NaN` in float16 on GTX-class cards, and libsndfile
writes NaN as a constant **−1.0** — a flat DC line. Correct duration, plausible
file size, no sound.

**Fix:** force `.float()` on `ema_model`, `model` and `vocoder`. Do not
"optimise" this back to half precision on a 16-series card.

**Method that found it:** measuring the samples, not listening. Every sample was
exactly −1.0. Guessing would never have found this.

> RTX 30-series and newer have usable tensor cores and run fp16 correctly —
> roughly a 2× speedup. This limitation is specific to GTX 16-series.

---

## 4. Text handling

The model reads characters. Anything it can't read, it mangles or skips.

**English words** — out of distribution. `translit.py` transliterates via the
CMU pronouncing dictionary (126k words), following how a word is *said* rather
than spelled: `buyer → बायर`, `Scientific → सायंटिफिक`, `IT → आय टी`. The
pronunciation dictionary always overrides it.

**Numbers** — barely present in training, so they come out wrong or get skipped.
`numerals.py` spells them out:

| input | spoken |
|---|---|
| `३०४` | तीनशे चार |
| `1994` | एकोणीसशे चौऱ्याण्णव *(year form)* |
| `2019` | दोन हजार एकोणीस *(not वीसशे)* |
| `MH 09 BT` | एम एच शून्य नऊ बी टी *(plate → digit-wise)* |
| `११:४५` | अकरा पंचेचाळीस |
| `150000` | एक लाख पन्नास हजार |

**Symbols** — the model has no character for them and silently drops them.
`50%` read as "पन्नास". Now expands `%`, `₹`/`Rs`, `°`, `&`, `×`, `=`, `@`.

**Stage directions** — `[वातावरणनिर्मिती ...]` and `*[सूचना: ...]*` were being read
aloud. Stripped by line and inline.

**Abbreviations** — `डॉ. अनन्या` split *after* `डॉ.`, so a story opened by saying
"डॉ." alone: heard as "do". Abbreviation periods are now hidden behind a
private-use character during chunking and restored before synthesis, because
Python's `re` has no variable-length lookbehind.

**Line breaks are the pause specification.** An early version ran
`re.sub(r"\s+", " ", text)` and destroyed every pause the author wrote. Never
collapse whitespace.

---

## 5. The pacing crisis

The largest single defect, and it was in F5-TTS, not our code.

F5-TTS decides how long a line may take **before generating anything**, from its
UTF-8 byte count (`f5_tts/infer/utils_infer.py`):

```python
duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)
```

For Devanagari that is close to meaningless. Every codepoint costs 3 bytes
whether or not it takes time to say — matras, anusvara and the halant all cost
3 bytes, and **the halant makes a word shorter** by deleting the inherent vowel.

| word | bytes | syllables | bytes/syllable |
|---|---|---|---|
| `कमल` | 9 | 3 | **3.0** |
| `शाळा` | 12 | 2 | 6.0 |
| `स्वप्न` | 18 | 2.6 | **6.9** |

Plain text gets less than half the time conjunct-heavy text gets.

**Measured on real stories: 103 of 103 lines under-allotted in one, 102 of 103
in another.** `"तर?"` was given **0.3 seconds**. `शाळा बंद होती.` was given 1.04 s
when it needs 1.65 — which is exactly why `शाळा` went missing from the output.

Starved chunks rush, race, run sentences together, clip final words, and past a
point **drop words entirely**.

**Fix — `pacing.py`.** Counts real Devanagari syllables (aksharas), weights long
vowels, nasals and conjuncts, adds time for punctuation, calibrates against the
reference clip's **measured** speaking rate, and passes F5-TTS an explicit
`fix_duration`.

Two details that matter:

- **Measure the reference the way F5-TTS does.** `preprocess_ref_audio_text`
  strips the silent edges and appends 50 ms — timing against the file on disk
  was timing against 5.60 s when 5.37 s was actually used.
- **The whole output inherits the reference clip's tempo.** A brisk clip makes
  every story brisk. The app now warns above ~7.5 moras/sec.

Audit any script with `tools/pacing_report.py`.

---

## 6. Chunking

Evolved through several wrong answers.

**Cap by duration, not characters.** Characters are a poor proxy for duration —
that is the whole point of §5.

**Long chunks slur.** Probe phrases of 1–2 s came out correct while story chunks
of 4 s median (15 s worst) did not.

**But splitting anywhere is worse.** Capping at 3.2 s by cutting at any
convenient word cut **35% of chunks mid-phrase** — `गणपतराव देशमुख यांना त्या`
rendered as its own utterance. Each fragment then gets sentence-final intonation
and a pause lands where no grammatical break exists.

**Final rule:** split at clause punctuation only — full stops first, then
commas. A sentence with nothing to split on stays whole even if it runs over.
Word-level cutting survives only above `hard_secs=9.0`, where F5-TTS would
otherwise split it itself and cross-fade a seam we cannot see.

Result on one story: mid-phrase cuts 35% → 4%, chunks 327 → 201, median duration
4.47 → 3.11 s.

**Short fragments rejoin.** `split_sentences` merges sentences under 26
characters so F5-TTS is never handed a tiny utterance, but the duration splitter
undid that at every full stop. It now rejoins.

**Single-pass chunks.** F5-TTS budgets ~22 s of audio per forward pass,
reference included, and **silently splits and cross-fades anything longer** —
the log says `Generating audio in 2 batches`. Those hidden seams caused slurring
and "broken sentences". Chunk size is capped so every chunk renders in one pass.

---

## 7. Edges and clipping

Three separate defects wearing the same clothes.

**1. Clipped final words.** A chunk still at full voice at its last moment was
cut off. `ends_mid_word()` detects it and re-renders. Measured at ~5% of chunks
before the fix.

**Threshold calibration matters.** Set by guess at −18 dB, it let audibly-cut
chunks through at −20.1 and −19.7 dB. A cleanly decaying ending measures −90 dB,
so there is a wide gap; **−25 dB** catches the near misses with no false
positives, verified against six chunks labelled by ear.

**2. The fade that caused what it prevented.** An unconditional 8 ms fade on
every chunk edge ate the first syllable of any chunk beginning on speech. Fades
now only run into material that is already quiet; a loud edge gets padded.

**3. Abrupt start and stop of the whole file.** Not truncation — no room.
`trim_silence` strips each chunk's own head and tail on purpose so the gaps
between them are exact, and the side effect was finished stories starting 125 ms
in and ending **0–160 ms after the last syllable**. One stopped dead at 0 ms.
Fixed with `lead_ms=350`, `tail_ms=900`.

**Do not disable trim to fix this.** Measured with `--no-trim`: each chunk keeps
an average 846 ms of head silence and 178 ms of tail, and the 250 ms pause
stacks on top — **a 1274 ms gap between every chunk**, which reads as word-by-word
delivery.

---

## 8. Seed variance

`F5TTS.infer` draws a fresh random seed per call when none is given:

```python
if seed is None:
    seed = random.randint(0, sys.maxsize)
```

Flow matching starts from noise, so the seed changes the result. Rendering one
sentence on ten seeds:

| roominess | clean takes |
|---|---|
| 1.30 | **4 / 10** |
| 1.45 | **9 / 10** |

A starved chunk rushes whichever word is hardest, and **which word that is
changes with the seed.** That is why the same word could be right in one render
and wrong in the next, and why per-word investigation kept finding the previous
word rendering fine.

Two consequences:

- **Roominess 1.35** (clamping to the 1.45 ceiling) is the default.
- **Pin the seed.** `--seed 7 --lock-seed` uses one seed for every chunk. A
  reproducible run is one where a bad chunk can be deliberately re-rendered.

**Do not over-allot.** At 1.8× the estimate the model does not leave the surplus
as silence — it **fills it with fluent nonsense** (an ASR transcribed one such
render as Kannada). `MAX_ROOMINESS = 1.45` is a hard ceiling for that reason.

---

## 9. Catastrophic forgetting — the big one

**The single most important finding, and it came from the user pushing back.**

**Symptom:** after fine-tuning, the voice was right and the Marathi was wrong.
`थांबलं` read as `थांबला`. `लायब्ररी` mangled. `कोल्हापूर` with a `ळ`. `अनन्या` as
"anya". Scattered, inconsistent, a different word each render.

**The wrong diagnosis (mine).** I measured that `थांबलं` appeared **twice** in the
training transcripts and `लायब्ररी` four times, concluded the model had never
learned them, and said the fix was recording more hours. A day of pipeline work
preceded that — the pipeline fixes were real, but none of them touched this.

**The correction (the user's).** *"I have trained a model which was already
trained for Marathi. It was just an exercise to give my voice."*

That reframes everything. IndicF5 already knew Marathi from hundreds of hours.
The question is not why the model never learned these words — it is **why it
forgot them.** 80 epochs over 1.26 hours is catastrophic forgetting.

**Evidence that supported the reframe, not the original theory:**

- `ळ` appears in the training transcripts at **7.22 per 1000** Devanagari
  characters vs 9.34 in the story scripts — near-natural frequency, so Whisper
  had *not* dropped it.
- Every `ळ`-word in the failing story had been seen in training — 20 of 20
  distinct forms.
- The `-लं` / `-ला` ratio was **0.48 in training vs 0.50 in the stories** — no
  distributional skew.
- Isolated probes of the failing words came back **correct** on all three
  checkpoints.

**The fix — weight interpolation, ~10 minutes, no retraining:**

```
merged = alpha * finetuned + (1 - alpha) * base
```

Speaker identity lives in a small, low-rank part of the weights; general
pronunciation is spread across all of them. A partial blend keeps the voice and
recovers the language. `tools/merge_ckpt.py`, `--sweep` to audition.

**alpha 0.50 is the answer here.** Listening verdict: *"all cha, la and kolhapur
pronounce were on point"*, voice *"90% on point which is ok"*. Saved as
`model_voice_merged50.pt`, now the default. 0.70 and 0.85 still mispronounced;
0.30 lost too much voice. Blends at 0.55/0.60/0.65 kept for future tuning.

### Converting the base for the merge

The base ships as `model.safetensors` with keys prefixed
`ema_model._orig_mod.` plus a bundled `vocoder.*`.

**Strip only `_orig_mod.` and drop the vocoder. Keep `ema_model.`.**

```python
out = {k.replace("_orig_mod.", ""): v for k, v in base.items()
       if not k.startswith("vocoder.")}
```

Then all **364 tensors align by name and shape**. Stripping `ema_model.` as well
gives **zero** matches — that was the first attempt.

Needs `hf auth login` and accepted IndicF5 terms.

### If retraining ever happens

The lesson is **not** "more data". It is fewer epochs, a much lower learning
rate, or LoRA so the base weights barely move. Forgetting was the cost, not
data volume.

---

## 10. Prosody chaining

**The fix for "sounds like 200 separate recordings".**

F5-TTS is an infilling model. It builds `ref_text + gen_text` as ONE utterance,
generates the whole thing conditioned on `ref_audio`, then slices the reference
portion off (`f5_tts/infer/utils_infer.py`):

```python
text_list = [ref_text + gen_text]
generated = model_obj.sample(cond=audio, text=final_text_list, ...)
generated = generated[:, ref_audio_len:, :]
```

So **the output is a continuation of whatever audio it is handed.** The pipeline
was handing it the same 5-second clip for all 200 chunks of a story, which
restarted the intonation, energy and tempo on every single one. The model has a
continuation mechanism and we were resetting it 200 times.

### Two versions, and why the first one failed

**Pure chaining** — replace the reference with the previous chunk. Flow improved
audibly. Voice fidelity did not survive it.

The obvious theory was compounding drift, and it was **wrong**. Measured over a
213-chunk story, distance from the reference by quarter:

```
CHAINED    0.012  0.015  0.015  0.016    drift +0.004
UNCHAINED  0.009  0.013  0.015  0.015    drift +0.005
```

The unchained run wanders just as much. Re-anchoring already stopped the
runaway. What chaining actually did was sit **consistently further out** — mean
0.015 against 0.013, worst 0.049 against 0.041. Not a voice sliding away; a
voice that is subtly not yours from the first chunk and never recovers.

(The first `drift_limit` was set at 0.06 on that wrong theory — above the 0.050
worst chunk ever observed, so it could never have fired. Dead code, shipped
confidently.)

**Anchored chaining** — concatenate instead of replacing:

```
reference = [the real clip] + [0.12s gap] + [previous chunk]
text      = [reference text] + [previous chunk text]
```

Genuine human audio stays in the conditioning window, so timbre is anchored to
a real recording while the previous chunk still supplies the prosodic run-up.
The real clip goes **first** because F5-TTS trims an over-long reference by
accumulating from the start — anything lost is the synthetic tail, never the
anchor. If the previous chunk will not fit under the 12 s ceiling, it falls back
to the plain reference rather than risking it.

Measured on the same passage, same seed:

| | voice-distance from reference |
|---|---|
| no chaining | 0.0116 |
| pure chaining | **0.0123** |
| **anchored** | **0.0115** |

Anchored matches unchained fidelity while still chaining.

### Guards

- re-anchor every `chain_reanchor` chunks (4 in `app.py`, 6 from `run_queue.py`),
  and at every paragraph and line
- re-anchor on any chunk whose fingerprint lands beyond `drift_limit` (0.045),
  set just above the 0.041 unchained generation reaches naturally so normal
  variation does not trip it
- never chain from a take that came back NaN, silent or implausibly short
- clear F5-TTS's reference cache periodically — it keys on md5 and never
  evicts, and chaining hands it a new file every chunk

`--no-chain` disables it. `_chain/prev_NNNN.wav` is written only when a chunk
actually chains, so that folder is the evidence it is working; it is deleted
once the story is assembled.

---

## 11. Acoustic post-processing

Vocos decodes accurate speech that is also **dry and thin** — flat low end, a
brittle top, and volume that jumps between a whispered line and a shouted one.
Commercial systems put a mastering chain after the vocoder. `warmth.py` is that
chain, kept gentle:

| stage | what | why |
|---|---|---|
| warmth | +2.5 dB low shelf at 200 Hz | chest resonance |
| de-harsh | 2nd-order roll-off from 11 kHz | diffusion hiss and sibilance |
| compress | 2:1 soft-knee | quiet and loud lines sit closer |

The roll-off is deliberately **gentle**. A 4th-order brick wall reads as
muffled; a 2nd-order slope takes the edge off and leaves air.

It runs at the end of every render and **always writes the unprocessed mix
alongside** as `<name>_raw.wav`, so it is reversible and an A/B never needs a
re-render. `--no-warmth` skips it, and a failure in the DSP logs and passes the
raw audio through rather than costing a story.

The module lives at the project root because the pipeline uses it;
`tools/warmth.py` is a thin CLI over the same code, so the two cannot drift
apart.

---

## 12. Gradio traps

Building the review UI cost four rounds, three of them avoidable.

**The audio element is a decoy.** With a story loaded and playing, Gradio's
`<audio>` tag reports `src: ""`, `readyState: 0`, `duration: null`,
`currentTime: 0`. Playback runs through a WaveSurfer instance not reachable from
the page. Reading `currentTime` always returns 0.

**The position is rendered as text:**

```html
<time>0:19</time>   <time>10:47</time>
```

Current, then duration. Group the pairs and take the one with the longest
duration — several players share the page.

**Dataframe round-trips fail silently.** Passing a `gr.Dataframe` in and out of
handlers, and through a `js=` function, produced no error and no result. Flags
now live in a JSON file on disk; only strings and numbers cross the boundary.
This also survives a page refresh.

**`fn=None` with `js=` did not bind.** Use an identity function with the JS
supplying the value.

**Parse timestamps the way people write them.** `0.19` means nineteen seconds,
not 0.19 seconds. Both `.` and `:` are minute separators here.

**Other Gradio notes:** `allowed_paths` must include every served directory;
Gradio 6 returns a pandas DataFrame from `gr.Dataframe`; `col_count` is now
`column_count`.

---

## 13. Hardware

The GTX 1650 in this machine **idles at 79 °C** and throttles within minutes of
load. Normal idle for that card is 35–45 °C.

| condition | per-chunk render |
|---|---|
| cool card | **22 s** |
| throttled (90 °C) | **65 s** |

A story that should take 40 minutes takes two hours. Eight stories took ~20
hours instead of ~6. **The 3× penalty is thermal, not computational.**

A between-story cooldown was added (`--cool-below`, `--cool-max-min`) and
**measured not to help on this machine** — eight idle minutes shed zero degrees.
A card that cannot cool at idle has a dust or paste problem, and compressed air
through the heatsink is worth more than any setting change here.

Other hardware notes:

- **Never store models or venv on FAT32 / removable media.** The 4 GB file cap
  rejects checkpoints, and the drive disconnected mid-run more than once.
- 4 GB VRAM fits one model. Do not run the UI and a batch render together.
- On Windows, `venv\Scripts\python.exe` spawns the base interpreter as a child —
  two `python.exe` processes for one job is normal, not a duplicate run.

---

## 14. Tools

| tool | what it answers |
|---|---|
| `tools/merge_ckpt.py` | blend a fine-tune back toward its base (§9) |
| `tools/pacing_report.py` | which lines of a script are being starved of time |
| `tools/analyze_chunks.py` | which chunks got cut off, and how long they ran |
| `tools/probe.py` | can the model say this word at all — `--inflected`, `--name`, `--failing`, `--variants` |
| `tools/seed_test.py` | is this failure seed luck — one text, N seeds |
| `tools/context_test.py` | is it the settings or the surroundings |
| `tools/tail_variants.py` | which stage is eating the last word |
| `tools/check_words.py` | ASR check that expected words survived |
| `tools/repair.py` | re-render named chunks and rebuild the story |
| `tools/make_recording_script.py` | targeted script for a future recording session |
| `review.py` | listen, flag by timestamp, repair — port 7861 |
| `progress.ps1` / `mac/progress.sh` | is the runner *actually* alive |

**`check_words.py` caveat:** it reported `आहे` missing from every variant, but
Whisper had heard `एकटाच आहे` as `एक ता चाहे` — ordinary Marathi liaison, not a
truncation. Read its "missing" column as a hint, never a verdict.

**`minimal_pair.py` caveat:** kept for the method, but its output was not
trustworthy here — the same `ळ`/`ल` contrast scored 0.55, 2.30 and 0.95 across
three word pairs. Listening is the better instrument at this scale.

---

## 15. Settings that matter

Defaults, and why.

| setting | value | reason |
|---|---|---|
| checkpoint | `model_voice_merged50.pt` | §9 |
| NFE | **32** | 16 is a draft; past ~40 the solver over-sharpens and adds buzzy artifacts |
| CFG | **2.0** | judged more natural than 2.2 by ear |
| roominess (`--pace`) | **1.35** → clamps to 1.45 | 4/10 → 9/10 clean takes (§8) |
| seed | **7, locked** | reproducibility (§8) |
| max seconds per chunk | **3.2** | long chunks slur (§6) |
| trim silence | **on** | off causes 1274 ms gaps (§7) |
| min pause | 250 ms | |
| full stop | 300 ms · line 350 · paragraph 800 | |
| lead / trail | **350 / 900 ms** | §7 |
| reference clip | **5–6 s** | every chunk re-synthesises it; a 9 s clip drops the single-pass cap from ~160 to ~118 chars |
| **chaining** | **anchored, on** | §10 — the fix for prosody restarting every chunk |
| chain re-anchor | 6 chunks · every paragraph · drift > 0.045 | bounds the fidelity cost |
| **warmth** | **on**, raw kept alongside | §11 |
| flow pause | **60 ms** | inside a split sentence. A comma is a breath, not a stop — 250 ms there made one sentence sound like two |
| max seconds per chunk | **8.0** | at 3.2, **26%** of chunks were comma-fragments |

**Reference clip transcripts must match word for word.** This affects quality
more than almost any other setting.

---

## 16. Diagnostic method

What actually worked, and what wasted time.

**Measure, don't guess.** The silent-audio bug was found by printing sample
values. The pacing bug was found by computing bytes per syllable. Both were
invisible to listening.

**Reproduce before fixing.** Three rounds went into the review UI's flag button
on three different wrong theories, because the DOM was never inspected. Fifteen
seconds of `document.querySelectorAll('audio')` would have found it first.

**Test the function, not the UI.** The flag logic was correct the whole time —
Gradio was discarding the result. Calling the function directly showed that
immediately.

**Calibrate thresholds against labelled data.** The clipped-edge detector was
set by guess and missed real faults by 2 dB. Six chunks labelled by ear fixed it
in one pass.

**When per-word investigation keeps finding the previous word fine, suspect the
model, not the plumbing.** The `ळ` → `खोल्या` → `थांबलं` → `वाचायची` chase was a
scattered error rate, which is what a damaged model looks like.

**Bit-identical renders can get opposite verdicts.** The same audio was judged
right in one listening session and wrong in another; neighbouring items shift
perception. Shuffle the order when auditioning.

**Do not write a liveness check with a command that may not exist.** A
graceful stop watcher called `pgrep -f run_queue.py`. `pgrep` is not in Git
Bash on Windows; the missing command exited non-zero, the "has it died?" test
read that as *yes*, and it killed a story at 73%. The replacement is a `STOP`
file checked inside the runner between stories - `Path.exists()` cannot be
misread, and the decision happens where it belongs.

**Configure your own stdout.** `run_queue.py` inherited cp1252 on Windows and
every story failed with `'charmap' codec can't encode characters`, which says
nothing about the real cause. It only worked because `overnight.bat` happened
to export `PYTHONIOENCODING`. Scripts that print Devanagari now call
`sys.stdout.reconfigure(encoding='utf-8')` themselves.

**Take the user's reframe seriously.** §9 exists because a stated assumption was
challenged, and the challenge was correct.

---

## 17. Still open

**Naturalness.** Prosody chaining (§10) addressed the worst of this — a chunk
now continues the previous one rather than restarting from a fixed clip. What
remains is that chaining is bounded: it re-anchors every few chunks and at every
paragraph, so a contour still cannot span a whole page. Long-range structure is
beyond what a 22-second conditioning window can hold.

**Anchored chaining is validated at short length, not long.** The three-way
measurement (0.0116 / 0.0123 / 0.0115) was on a 6-chunk passage where the gap
between chained and unchained was only 0.0007. A 213-chunk story is where a
residual offset would actually become audible. Directionally right, not proven.

**Residual error rate.** Even merged and well-paced, a small fraction of chunks
land badly. `review.py` + `repair.py` exist to catch those by ear rather than
re-render whole stories.

**Splicing was proposed and not built** — recording the few bad sentences
yourself and dropping the wav in, with the story rebuilding around it. That is
the shortest path to publishable audio, and every sentence recorded that way is
also training data for a future fine-tune.

**If retraining:** fewer epochs, lower learning rate, or LoRA. See §9.
`recording_script.txt` targets the weakest sounds if more data is wanted, but it
is no longer the priority it appeared to be.
