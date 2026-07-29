# Changelog

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
