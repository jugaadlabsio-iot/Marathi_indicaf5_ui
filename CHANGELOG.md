# Changelog

## [v1.2.0] — 2026-07-30 · Pacing

v1.1.0 fixed how chunks were *joined*. This fixes how long each chunk is
allowed to *take*, which turned out to be the cause of most of what was still
wrong: rushing, racing starts, sentences running together, clipped final
words, and words disappearing entirely.

### The bug, in one line

F5-TTS budgets speech time by **UTF-8 byte count**:

```python
duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)
```

In Devanagari that is close to meaningless. Every codepoint costs 3 bytes
whether or not it takes time to say — matras, anusvara and the halant all cost
3 bytes each, and the halant actually makes a word *shorter* by deleting the
inherent vowel. So:

| word | bytes | syllables | bytes per syllable |
|---|---|---|---|
| `कमल` | 9 | 3 | **3.0** |
| `शाळा` | 12 | 2 | 6.0 |
| `स्वप्न` | 18 | 2.6 | **6.9** |

A line of plain consonants gets less than half the time a conjunct-heavy line
gets. Running the audit over two of the twelve QA stories: **103 of 103 lines
under-allotted in one, 102 of 103 in the other.** `"तर?"` was given 0.3
seconds. `शाळा बंद होती.` was given 1.04 s when it needs 1.65 s — which is
exactly why `शाळा` went missing.

### Fixed

- **New `pacing.py`** counts actual Devanagari syllables (aksharas), weights
  long vowels, nasals and conjuncts, adds real time for punctuation, and
  calibrates against the reference clip's own **measured** speaking rate. The
  result is passed to F5-TTS as an explicit `fix_duration`. Toggle:
  *Budget duration by syllables*.
- **Reference length is now measured correctly.** F5-TTS does not use the wav
  on disk — `preprocess_ref_audio_text` strips the silent edges and appends
  50 ms. Timing against the file on disk was timing against the wrong number
  (5.60 s vs the 5.37 s actually used).
- **One sentence per chunk.** Packing several sentences into one chunk left the
  model to decide how long to rest at a full stop, and under time pressure it
  decided *not to*. Each sentence now gets its own start, end, and a real
  pause. Sentences under 26 characters are merged with a neighbour, because
  F5-TTS is unstable on tiny utterances.
- **Edge fades no longer clip word onsets.** v1.1.0 faded every chunk edge
  unconditionally, which ate the first syllable of any chunk that began on
  speech — a fix that caused the thing it was meant to prevent. Fades now only
  run into material that is already quiet; a loud edge gets padded instead.
- **Automatic re-roll on suspected dropped words.** If a chunk uses under 58 %
  of its planned time, it is regenerated once on a different seed and the
  fuller take is kept.
- **Warning when the reference clip is brisk.** The entire output inherits the
  reference's tempo. Above ~7.5 moras/sec the log now says so and suggests
  raising Roominess or recording a calmer clip.

### Added

- **Roominess** slider (0.85–1.35) — extra time on top of the estimate. Raise
  it if delivery still feels hurried; lower it if lines drawl.
- **Pause at a full stop** slider (default 260 ms).
- **`tools/pacing_report.py`** — audit any script and see which lines were
  being starved, worst first. Run it against a story you already disliked; the
  flagged lines should be the ones that sounded wrong.
- `seed` for reproducible generation (`--seed` in the queue runner).
- Queue runner flags: `--pace`, `--sent-pause`, `--byte-duration`,
  `--no-sentence-split`, `--seed`.

### Known, and not fixable here

`ळ` read as `ल`, `थांबले` read as `थांबला`, and mispronounced proper nouns like
`कोल्हापूर` are **training-data** problems, not pipeline problems. The
fine-tuning transcripts were produced by Whisper large-v3, which misspells
Marathi — it writes `ल` for `ळ` and normalises verb endings. The model learned
whatever the transcripts said. No inference setting fixes this. The options are
pronunciation-dictionary respellings, trying `model_8000.pt` (less overfit to
the bad transcripts than `model_last.pt`), or correcting `metadata.csv` and
retraining.

---

## [v1.1.0] — 2026-07-29 · Audio quality

Everything here came from listening to twelve generated stories and writing down
what was wrong. Each entry names the symptom it fixes.

### Fixed

- **Broken sentences, cobbled words, buzzy artifacts at high quality.**
  F5-TTS budgets roughly **22 seconds of audio per forward pass**, reference clip
  included, and silently splits and cross-fades anything longer — the log said
  `Generating audio in 2 batches` for a single chunk. Those hidden seams were the
  problem. Chunk size is now derived from the reference clip
  (`single_batch_limit()`), so every chunk renders in one pass and every join is
  one we control, with a real pause. On by default; `--allow-splits` restores the
  old behaviour.
- **Aggressive clipping at the end of lines.** Silence trim was too eager:
  threshold −40 → **−55 dBFS**, keep 40 → **120 ms**. Quiet trailing consonants
  were falling under the old floor and being cut off.
- **Abrupt starts and clicks at short pauses.** Every chunk now gets an **8 ms
  fade** at both edges. A hard cut between two waveforms is a step discontinuity,
  and that is what you hear.
- **Numbers, dates and plates read wrong or skipped.** New `numerals.py` spells
  digits out as Marathi words before synthesis — digits are barely present in the
  training data. Years, lakhs, plates and times each get the reading a narrator
  would actually use. On by default; `--keep-digits` turns it off.

  | input | spoken as |
  |---|---|
  | `३०४` | तीनशे चार |
  | `1994` | एकोणीसशे चौऱ्याण्णव |
  | `2019` | दोन हजार एकोणीस |
  | `MH 09 BT` | एम एच शून्य नऊ बी टी |
  | `११:४५` | अकरा पंचेचाळीस |
  | `150000` | एक लाख पन्नास हजार |

### Added

- **Truncation warning** — logs when a chunk's audio is far shorter than its text
  would imply, so a silently cut sentence doesn't reach the final mix unnoticed.
- Two new toggles under *Script handling*: **Read numbers as Marathi words** and
  **Force single-pass chunks**. Both default on.

### Changed

- **NFE guidance corrected.** 16 draft · **32 final**. Past ~40 the solver
  over-sharpens and adds the buzzy artifacts previously reported at "studio
  quality" — more steps is *not* more quality here.

### Notes

English words are deliberately left to the pronunciation dictionary. Your
overrides always run first, before number expansion and before the automatic
CMU-based transliteration.

---

## [v1.0.0] — 2026-07-28 · First working studio

- Gradio UI on `localhost` — paste a script, press Generate.
- Overnight batch queue; a failing story never stops the run.
- Automatic English → Devanagari via the CMU pronouncing dictionary (126k words).
- Pronunciation dictionary with word-level overrides.
- Stage directions stripped rather than narrated.
- Layout-driven pacing: line breaks and blank lines become real pauses.
- Reference clip manager with per-clip transcripts.
- Crash-resilient: every chunk written to disk as it finishes.
- macOS (Apple Silicon) support via `mac/setup_mac.sh`.
- **float32 enforced on CUDA** — in float16 the model emits NaN, and libsndfile
  writes NaN as a constant −1.0, i.e. a flat DC file that sounds like silence.
