# Writing stories this pipeline reads well

Paste the block below into any LLM when you want a new Marathi horror story.
Every number in it is measured against this setup, not guessed — the reasoning
behind each rule is in the second half of this file.

---

## The prompt

> Write a Marathi horror story (भयकथा) in the style of नारायण धारप × रत्नाकर
> मतकरी. It will be narrated by a text-to-speech system, so follow these
> formatting rules exactly. They are technical requirements, not style
> preferences.
>
> **Output plain prose only.** No headings, no title line, no reading-time
> estimate, no horizontal rules, no production notes, no `*[सूचना: ...]*` audio
> cues, no markdown emphasis, no bullet lists, no tables. Every character in the
> file will be spoken aloud. If it isn't meant to be heard, don't write it.
>
> **Sentences: keep them under 80 Devanagari characters.** This is the single
> most important rule. A longer sentence gets cut at a comma and the two halves
> are narrated as separate utterances — the first half resolves its intonation
> as though the sentence ended. If an idea needs more room, use two sentences
> rather than one sentence with commas.
>
> **Avoid comma-lists of three or more items.** `जुनी वडाची, पिंपळाची, उंबराची
> झाडं…` is the worst case: it will be split mid-list and the join will sound
> broken. Two items are fine; three, split into separate sentences.
>
> **Paragraphs: 150–350 characters for narration.** A blank line is a real
> pause. Paragraphs of 140 characters or less get a short pause (320 ms),
> longer ones get a full pause (600 ms) — so use short paragraphs for dialogue
> and beats, longer ones for description.
>
> **Never put a single word or a two-word fragment alone in its own
> paragraph.** `का?` or `मग?` on its own line renders as a 0.4-second fragment
> that stops dead. Attach it to a speech tag or to the following line:
> `"का?" तो म्हणाला.`
>
> **Numbers: write them as Marathi words, never digits.** `सात शून्य दोन` for a
> room number, `एकोणीसशे बासष्ठ` for a year, `रात्रीचे बारा` for a time. Digits
> are read as cardinals, so a door number `७०२` becomes "सातशे दोन".
>
> **Acronyms: separate the letters with spaces.** `आय टी`, `जी पी एस`,
> `बी के सी`, `पी एच`. Written joined, they are read as words.
>
> **No Latin script anywhere.** English loanwords are fine in Devanagari
> (`लॅपटॉप`, `मोबाइल`, `ट्रेन`) — keep to common, familiar ones.
>
> **Dialogue: one speaker per paragraph**, plain `"…"` quotes, no nested
> quotes. Keep spoken lines short and natural.
>
> **Punctuation carries the performance.** `.` `?` `!` `।` end sentences;
> `—` gives a beat; `…` gives a long dramatic pause; `,` a small breath. Use
> them deliberately. Avoid parentheses and semicolons.
>
> Target length: [X] minutes of narration ≈ [X × 700] characters.

---

## Why each rule exists

**80 characters per sentence.** With the current reference clip (8.26
moras/sec) and roominess 1.20, an 8-second chunk holds about 80 Devanagari
characters. Anything longer is split by `split_by_duration()` at the nearest
comma. The model then renders each half as a complete utterance, because it has
no idea the sentence continues. This produced the single clearest defect anyone
reported: `जुनी वडाची, पिंपळाची,` ending on a falling contour, then `उंबराची
झाडं…` starting cold. Finished stories average **52 characters per chunk**, and
the longest chunk observed was 120.

**150–350 characters per paragraph.** `SHORT_PARA_CHARS = 140` is the cutoff
between a 320 ms and a 600 ms pause. Before that distinction existed, every
blank line got 800 ms *and* reset the voice, and a dialogue exchange written one
line per paragraph read as a series of unrelated statements. Paragraph density
tracked perceived quality almost exactly across six stories: 174 characters per
paragraph in the best-rated story, 83 in the worst.

**No one-word paragraphs.** `MIN_SENT_CHARS = 26` merges short sentences with a
neighbour, but only *within a line*. A single word alone between blank lines has
nothing to merge with, so it renders as a fragment of 0.4 s that stops without
decay. Three separate repair strategies were tried on `का?` — more duration,
seed retries, and re-rendering with trailing context — and none of them worked.
Not writing them is the only fix.

**Numbers as words.** `numerals.py` expands digits to Marathi cardinals, which
is right for counting and wrong for identifiers. `७०२` as an office number
became "सातशे दोन".

**Acronyms spaced.** A joined string like `आयटी` is a valid Devanagari word to
the model, so it gets pronounced as one.

**No stage directions or headings.** These files are usually written as
documents. `tools/extract_stories.py` strips headings, reading-time lines,
rules and `*[सूचना: …]*` cues — but only if they match its patterns. Text that
was never written is text that can never leak into the audio.

**Length estimate.** Roughly **700 characters per minute** of finished
narration, measured across eight stories.

---

## What this cannot fix

Individual words may still be mispronounced — `ळ` versus `ल`, and dental versus
palatal `च`/`ज`. That is a property of the model, not the text. When you hear
one, add it to `pronunciation.json`; it then applies to every future story.
