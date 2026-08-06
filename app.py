# -*- coding: utf-8 -*-
"""
Marathi Story Voice - local web UI for your fine-tuned IndicF5 voice.

Windows + NVIDIA -> CUDA (float32 forced; fp16 gives NaN / silent audio)
Mac mini M4      -> Apple Silicon MPS
otherwise        -> CPU

    python app.py     ->  http://127.0.0.1:7860
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import sys
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import random
import re
import shutil
import time
import platform
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch

# These live beside app.py. A bare `import pacing` resolves against sys.path,
# which is the CALLER's - so loading app.py by path from tools/ put only
# tools/ on it and every one of these silently became None. The symptoms were
# not obviously import-related: pacing=None disables duration budgeting, so
# the audio just sounds rushed; numerals=None reads digits as digits; and
# warmth=None skips post-processing. Every A/B rendered through tools/ was
# affected, which invalidated both the paragraph comparison and the merge
# sweep before anyone noticed. run_queue.py sits in this directory so real
# renders were never hit - which is exactly why it went unseen for so long.
# Not BASE - that is defined below and honours MTTS_HOME, whereas these
# modules always sit beside this file.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_MISSING = []
try:
    import translit                      # English -> Devanagari (CMU based)
except Exception as _e:
    translit = None
    _MISSING.append(f"translit ({_e})")
try:
    import numerals                      # digits -> Marathi words
except Exception as _e:
    numerals = None
    _MISSING.append(f"numerals ({_e})")
try:
    import pacing                        # how long should this text take to say
except Exception as _e:
    pacing = None
    _MISSING.append(f"pacing ({_e})")
try:
    import warmth                        # post-vocoder EQ + compression
except Exception as _e:
    warmth = None
    _MISSING.append(f"warmth ({_e})")
if _MISSING:
    print("*** app.py could not import its own modules: " + ", ".join(_MISSING) +
          "\n*** Output will be degraded (rushed pacing / unspoken digits).",
          file=sys.stderr, flush=True)

# ---------------------------------------------------------------- paths ----
BASE = Path(os.environ.get("MTTS_HOME", Path(__file__).resolve().parent))
DEFAULT_MODELS = Path(r"C:\marathi_tts_models") if os.name == "nt" else BASE / "models"
MODELS_DIR = Path(os.environ.get("MTTS_MODELS", DEFAULT_MODELS))
REF_DIR = Path(os.environ.get("MTTS_REFS", BASE / "ref"))
OUT_DIR = Path(os.environ.get("MTTS_OUT", BASE / "out"))
PARTS_DIR = OUT_DIR / "parts"
DICT_FILE = BASE / "pronunciation.json"
for d in (REF_DIR, OUT_DIR, PARTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()


def device_label():
    if DEVICE == "cuda":
        return f"CUDA · {torch.cuda.get_device_name(0)}"
    if DEVICE == "mps":
        return f"Apple Silicon MPS · {platform.machine()}"
    return f"CPU · {platform.machine()}"


def list_ckpts():
    return sorted(str(p) for p in MODELS_DIR.glob("*.pt")) if MODELS_DIR.exists() else []


def list_refs():
    return sorted(str(p) for p in REF_DIR.glob("*.wav")
                  if (p.with_suffix(".txt")).exists() or True)


def ref_text_for(path):
    if not path:
        return ""
    side = Path(path).with_suffix(".txt")
    return side.read_text(encoding="utf-8").strip() if side.exists() else ""


# ------------------------------------------------- pronunciation dictionary -
def load_dict():
    if DICT_FILE.exists():
        try:
            return json.loads(DICT_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_dict(d):
    DICT_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def dict_to_rows(d):
    return [[k, v] for k, v in sorted(d.items())]


def rows_to_dict(rows):
    """Accepts a list-of-lists or the pandas DataFrame gradio hands back."""
    out = {}
    if rows is None:
        return out
    if hasattr(rows, "values"):          # pandas DataFrame
        rows = rows.values.tolist()
    for row in rows:
        if row is None or len(row) < 2:
            continue
        k = str(row[0] or "").strip()
        v = str(row[1] or "").strip()
        if k and v and k.lower() != "nan" and v.lower() != "nan":
            out[k] = v
    return out


# --- English / Latin-script handling ---------------------------------------
# IndicF5 is trained on Indic script. Latin words are out of distribution and
# come out mangled or skipped, so the fix is to respell them in Devanagari.
# letters plus INTERNAL apostrophes/hyphens only - a trailing '.' is sentence
# punctuation and must survive, or chunking and pauses break
LATIN_RE = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")

# Words that actually appear in these horror scripts, plus common acronyms.
# Acronyms are spelled letter-by-letter the way a Marathi narrator says them.
BUILTIN_LATIN = {
    "buyer": "बायर", "Buyer": "बायर",
    "rational": "रॅशनल", "Rational": "रॅशनल",
    "scientific": "सायंटिफिक", "Scientific": "सायंटिफिक",
    "Netflix": "नेटफ्लिक्स", "netflix": "नेटफ्लिक्स",
    "Instagram": "इन्स्टाग्राम", "WhatsApp": "व्हॉट्सॲप",
    "Reels": "रील्स", "reels": "रील्स", "caption": "कॅप्शन",
    "camping": "कॅम्पिंग", "trekking": "ट्रेकिंग",
    "IT": "आय टी", "EMI": "ई एम आय", "OYO": "ओयो",
    "ST": "एस टी", "RTO": "आर टी ओ", "MH": "एम एच",
    "AC": "ए सी", "TV": "टी व्ही", "GPS": "जी पी एस",
    "OPD": "ओ पी डी", "MBBS": "एम बी बी एस", "PHC": "पी एच सी",
    "app": "ॲप", "online": "ऑनलाइन", "flat": "फ्लॅट",
    "hotel": "हॉटेल", "room": "रूम", "highway": "हायवे",
    "software": "सॉफ्टवेअर", "engineer": "इंजिनिअर",
    "office": "ऑफिस", "mobile": "मोबाइल", "torch": "टॉर्च",
}


def find_latin_words(text):
    """Unique Latin-script words in the script, most frequent first."""
    counts = {}
    for w in LATIN_RE.findall(text or ""):
        w = w.strip(".-'")
        if len(w) < 1:
            continue
        counts[w] = counts.get(w, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))


def scan_english(script, rows):
    """Add every English word found to the dictionary table, pre-filling the
    ones we know. Blank suggestions are for you to fill in."""
    d = rows_to_dict(rows)
    found = find_latin_words(script)
    if not found:
        return (dict_to_rows(d) or [["", ""]],
                "No English words found in the script.")
    added, known = 0, 0
    for word, _n in found:
        if word in d:
            continue
        suggestion = BUILTIN_LATIN.get(word) or BUILTIN_LATIN.get(word.lower()) or ""
        if suggestion:
            known += 1
        d[word] = suggestion or ""
        added += 1
    rows_out = [[k, v] for k, v in sorted(d.items())]
    missing = sum(1 for _k, v in rows_out if not v)
    msg = (f"Found {len(found)} distinct English word(s); added {added} new "
           f"({known} auto-filled). {missing} still need a Devanagari spelling - "
           f"blank entries are ignored at generation time.")
    return rows_out or [["", ""]], msg


def autofill_blanks(rows):
    """Fill every empty replacement with an automatic transliteration."""
    d = rows_to_dict(rows)
    blanks = []
    if hasattr(rows, "values"):
        rows = rows.values.tolist()
    for row in (rows or []):
        if row is None or len(row) < 2:
            continue
        k = str(row[0] or "").strip()
        v = str(row[1] or "").strip()
        if k and (not v or v.lower() == "nan") and k not in d:
            blanks.append(k)
    filled = 0
    for k in blanks:
        got = BUILTIN_LATIN.get(k) or BUILTIN_LATIN.get(k.lower()) \
            or (translit.transliterate(k) if translit else "")
        if got:
            d[k] = got
            filled += 1
    return ([[k, v] for k, v in sorted(d.items())] or [["", ""]],
            f"Filled {filled} blank entry(ies) automatically. Edit any that sound wrong.")


def apply_dict(text, d):
    """Whole-word respelling. The model is character-based, so the only way to
    steer pronunciation is to spell the word the way you want it said."""
    if not d:
        return text
    for word in sorted(d, key=len, reverse=True):     # longest first
        text = re.sub(rf"(?<![\w\u0900-\u097F]){re.escape(word)}(?![\w\u0900-\u097F])",
                      d[word], text)
    return text


def auto_translit_text(text):
    """Convert any remaining Latin-script word to Devanagari automatically."""
    if translit is None:
        return text

    def repl(mo):
        w = mo.group(0)
        got = BUILTIN_LATIN.get(w) or BUILTIN_LATIN.get(w.lower()) \
            or translit.transliterate(w)
        return got or w

    return LATIN_RE.sub(repl, text)


# --- stage directions -------------------------------------------------------
# Production notes belong to you, not to the narrator. Without this the model
# cheerfully reads "[वातावरणनिर्मिती बदल - गावातला रस्ता]" out loud.
_SD_LINE = re.compile(r"^\s*\*?\s*\[.*?\]\s*\*?\s*$")      # whole line is a cue
_SD_INLINE = re.compile(r"\*?\[[^\]\n]*\]\*?")             # cue inside a line


def strip_stage_directions(text):
    out = []
    for line in (text or "").split("\n"):
        if _SD_LINE.match(line):
            continue                                        # drop the line
        cleaned = _SD_INLINE.sub(" ", line)                 # drop inline cues
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).rstrip()
        # a line that was only a cue collapses to nothing - keep it dropped,
        # but preserve genuinely blank lines, they are your paragraph breaks
        if cleaned.strip() or not line.strip():
            out.append(cleaned)
    return "\n".join(out)


def fade_edges(wav, sr, ms=6):
    """Fade only INTO material that is already quiet.

    A hard cut between two waveforms is a step discontinuity - that is the
    click at a pause. But an unconditional fade is worse than the click: where
    a chunk begins on speech it eats the onset, which is a word getting
    clipped. So where the edge is loud we pad a few ms of silence and fade
    across the padding instead of across the speech.
    """
    n = max(2, int(sr * ms / 1000))
    if wav.size < 4 * n:
        return wav
    out = wav
    if np.abs(out[:n]).max() > 0.02:
        out = np.concatenate([np.zeros(n, np.float32), out])
    if np.abs(out[-n:]).max() > 0.02:
        out = np.concatenate([out, np.zeros(n, np.float32)])
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = out.copy()
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def starts_mid_word(wav, sr, window_ms=40.0, thresh_db=-25.0):
    """Was the opening syllable sliced off?

    F5-TTS generates [reference + target] as one utterance and then discards
    the first `ref_audio_len` frames. When the model begins the target a shade
    early it lands inside the discarded region and the first word loses its
    onset. Speech that starts naturally eases in; audio already at full voice
    in its first 40 ms began before the file did.

    It shows up worst on the opening chunk of a story, where there is no
    preceding audio to hide it - "the name the story starts with" getting cut.
    """
    n = int(sr * window_ms / 1000)
    if wav.size < 2 * n:
        return False
    peak = float(np.abs(wav).max())
    if peak < 1e-6:
        return False
    head = float(np.sqrt((wav[:n] ** 2).mean()))
    return 20 * np.log10(max(head, 1e-9) / peak) > thresh_db


def ends_mid_word(wav, sr, window_ms=40.0, thresh_db=-25.0):
    """Was the model still talking when its allotted time ran out?

    Speech that finishes naturally trails off. Speech still close to the
    chunk's peak at the very last moment was cut - that is the clipped final
    word, and unlike 'sounds wrong' it is measurable.

    The threshold was -18 dB and let two audibly-cut chunks through at -20.1
    and -19.7 dB. A cleanly decaying ending measures around -90 dB, so there
    is a wide gap to sit in; -25 dB catches the near misses without flagging
    healthy chunks.
    """
    n = int(sr * window_ms / 1000)
    if wav.size < 2 * n:
        return False
    peak = float(np.abs(wav).max())
    if peak < 1e-6:
        return False
    tail = float(np.sqrt((wav[-n:] ** 2).mean()))
    return 20 * np.log10(max(tail, 1e-9) / peak) > thresh_db


def trim_silence(wav, sr, thresh_db=-55.0, keep_ms=120):
    """Trim leading/trailing silence ourselves.

    F5-TTS's own remove_silence needs ffmpeg via pydub, which is not installed
    here. Doing it in numpy is dependency-free and, more importantly, makes the
    pauses exact: the model's own ragged head/tail silence is removed and the
    gaps you asked for are inserted instead.
    """
    if wav.size == 0:
        return wav
    fl = max(1, int(0.01 * sr))
    n = len(wav) // fl
    if n < 2:
        return wav
    frames = wav[:n * fl].reshape(n, fl)
    rms = np.sqrt((frames ** 2).mean(1) + 1e-12)
    peak = rms.max() + 1e-12
    loud = 20 * np.log10(rms / peak) > thresh_db
    if not loud.any():
        return wav
    first, last = int(np.argmax(loud)), int(len(loud) - np.argmax(loud[::-1]))
    pad = int(keep_ms / 10)
    a = max(0, (first - pad)) * fl
    b = min(len(wav), (last + pad) * fl)
    return wav[a:b]


def prepare_text(text, d, use_dict, auto, drop_directions=True, expand_nums=True):
    """Your dictionary wins; then numbers; then leftover Latin words."""
    if drop_directions:
        text = strip_stage_directions(text)
    if use_dict:
        text = apply_dict(text, d)          # your overrides run first, always
    if expand_nums and numerals is not None:
        text = numerals.expand_numbers(text)
    if auto:
        text = auto_translit_text(text)
    return text


# ------------------------------------------------------------- the model ---
_cache = {"key": None, "tts": None}


def _cast(tts, fn):
    for name in ("ema_model", "model", "vocoder"):
        obj = getattr(tts, name, None)
        if obj is not None and hasattr(obj, fn):
            setattr(tts, name, getattr(obj, fn)())
    return tts


def half_is_safe(tts, device):
    """Does this device produce real audio in half precision, or NaN?

    On GTX 16-series cards it produces NaN, which libsndfile writes as a
    constant -1.0 - a file of the right length that plays as silence. That is
    where the blanket .float() came from. But it is a limitation of those
    cards, not of the model: Apple Silicon and RTX cards run fp16 correctly and
    roughly twice as fast, so forcing fp32 everywhere taxes them for a bug they
    do not have. Rather than guess by device name, try it and look at the
    numbers.
    """
    ref = next((r for r in list_refs() if ref_text_for(r)), None)
    if ref is None:
        return False
    try:
        _cast(tts, "half")
        wav, _sr, _ = tts.infer(ref_file=ref, ref_text=ref_text_for(ref).strip(),
                                gen_text="चाचणी.", nfe_step=8, remove_silence=False)
        wav = np.asarray(wav, dtype=np.float32)
        ok = wav.size > 0 and not np.isnan(wav).any() and float(np.abs(wav).max()) > 1e-4
    except Exception:
        ok = False
    if not ok:
        _cast(tts, "float")
    return ok


def get_tts(ckpt, vocab, device, precision="auto"):
    """precision: 'auto' tests half once per device, 'fp32' forces full."""
    key = (ckpt, vocab, device, precision)
    if _cache["key"] == key:
        return _cache["tts"]
    from f5_tts.api import F5TTS
    tts = F5TTS(model="F5TTS_Base", ckpt_file=ckpt, vocab_file=vocab, device=device)
    _cast(tts, "float")

    want_half = precision == "half" or (
        precision == "auto" and device in ("mps", "cuda")
        and os.environ.get("MTTS_FORCE_FP32", "") != "1")
    if want_half:
        if half_is_safe(tts, device):
            print(f"  precision: half on {device} (verified, no NaN)")
        else:
            print(f"  precision: float32 on {device} "
                  f"(half produced NaN or silence - this is a GTX 16-series trait)")

    _cache.update(key=key, tts=tts)
    return tts


# -------------------------------------------------------------- chunking ---
# A period does not always end a sentence. "डॉ. अनन्या कुलकर्णी." was split
# after डॉ., so a story opened by saying "डॉ." on its own - heard as "do"
# instead of the doctor's title running into her name. The period is hidden
# behind a private-use character for the duration of chunking and put back
# before anything is spoken, because python's re has no variable-length
# lookbehind to express "not after an abbreviation".
_ABBREVS = ["डॉ", "श्री", "श्रीमती", "सौ", "कु", "प्रा", "पं", "स्व", "ता",
            "उदा", "वि", "मा", "अॅड", "इं",
            "Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr", "St", "No", "vs"]
_ABBR_DOT = re.compile(r"(?<![\wऀ-ॿ])(" + "|".join(_ABBREVS) + r")\.")
_ABBR_MARK = ""


def protect_abbrevs(text):
    return _ABBR_DOT.sub(lambda m: m.group(1) + _ABBR_MARK, text or "")


def restore_abbrevs(text):
    return (text or "").replace(_ABBR_MARK, ".")


_SENT = re.compile(r"(?<=[।\.\?\!])\s+")


def split_text(text, max_chars=400):
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    chunks, cur = [], ""
    for sent in _SENT.split(text):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > max_chars:
            for part in re.split(r"(?<=[,;—])\s+", sent):
                if len(cur) + len(part) + 1 <= max_chars:
                    cur = (cur + " " + part).strip()
                else:
                    if cur:
                        chunks.append(cur)
                    cur = part
            continue
        if len(cur) + len(sent) + 1 <= max_chars:
            cur = (cur + " " + sent).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = sent
    if cur:
        chunks.append(cur)
    return chunks


MIN_SENT_CHARS = 26        # below this a chunk is too short to synthesise well


def split_sentences(line, max_chars, min_chars=MIN_SENT_CHARS):
    """One item per sentence.

    Packing several sentences into one chunk is what made the narration run
    continuously: inside a chunk the model decides for itself how much to rest
    at a full stop, and when it is short of time it decides 'not at all'. One
    sentence per chunk means the gap between sentences is silence we insert,
    not something we hope for.

    Very short sentences are the opposite problem - F5-TTS is unstable on tiny
    utterances (it force-slows anything under 10 bytes), so anything under
    `min_chars` is glued to its neighbour rather than rendered alone.
    """
    sents = [s.strip() for s in _SENT.split(line) if s.strip()]
    merged = []
    for s in sents:
        if merged and (len(merged[-1]) < min_chars or len(s) < min_chars) \
                and len(merged[-1]) + len(s) + 1 <= max_chars:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    out = []
    for s in merged:
        parts = split_text(s, max_chars) if len(s) > max_chars else [s]
        for i, p in enumerate(parts):
            out.append((p, "chunk" if i < len(parts) - 1 else "sent"))
    return out


# A full stop is the safest seam there is, so it is tried first. It has to be
# listed separately because split_sentences() deliberately MERGES sentences
# shorter than MIN_SENT_CHARS, and without this a merged group like
# "सागर पवार. वय तीस. पुण्यात सॉफ्टवेअर कंपनीत काम." could never be broken up
# again - it rendered as a single 5.7s chunk at the very opening of a story.
_SENT_SEAM = re.compile(r"(?<=[।\.\?\!])\s+")
_CLAUSE = re.compile(r"(?<=[,;:—–])\s+")


def split_by_duration(items, rate, max_secs, roominess=1.08, hard_secs=9.0):
    """Shorten long chunks - but only where the language offers a seam.

    Measured on a real run: the model pronounces ळ, मी and च correctly in
    short phrases (1-2 s) and loses them in long ones (the stories ran to a
    4 s median and 15 s worst case). Capping by character count does not
    control this, because characters are a poor proxy for duration - which is
    the whole reason pacing.py exists. So cap by the estimate instead.

    The important restraint is WHERE. Splitting at any convenient word cost
    35% of chunks a mid-phrase cut on the first attempt ("गणपतराव देशमुख यांना
    त्या" as its own render): each fragment gets sentence-final intonation and
    a pause lands where no grammatical break exists, which sounds worse than
    the slurring it was meant to fix. So we split at clause punctuation only.
    A sentence with nothing to split on stays whole even if it runs over -
    long is better than butchered.

    `hard_secs` is the exception: past that a chunk risks F5-TTS's own
    internal splitting, whose seams we cannot control at all, so a word-level
    cut becomes the lesser evil.
    """
    if pacing is None or max_secs <= 0:
        return items

    def secs(t):
        return pacing.estimate_seconds(t, rate, roominess=roominess)

    def by_words(part):
        out, cur = [], ""
        for w in part.split(" "):
            trial = (cur + " " + w).strip()
            if cur and secs(trial) > max_secs:
                out.append(cur)
                cur = w
            else:
                cur = trial
        if cur:
            out.append(cur)
        return out

    def pack(parts):
        """Group clauses into as few pieces as possible, of even length.

        Two problems this replaces.

        Emitting one chunk per comma gave single-word utterances - '"चला,',
        'थंडगार,' - each with sentence-final intonation and a pause after it.
        A list like 'थंडगार, स्वच्छ, गोड पाणी' became three detached fragments.

        Filling greedily to max_secs then left a runt at the end: a 9 s
        sentence split 8 s + 1 s, and that 1 s tail is the "last word clubbed
        with the next sentence". So decide how many pieces are needed, then
        aim for even ones - two halves of 4.5 s read as one sentence taking a
        breath, where 8 s + 1 s reads as a sentence plus an orphan.
        """
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return []
        total = secs(" ".join(parts))
        if total <= max_secs:
            return [" ".join(parts)]

        import math
        n = max(2, math.ceil(total / max_secs))
        target = total / n                       # even, not full

        out, cur = [], ""
        for i, part in enumerate(parts):
            trial = (cur + " " + part).strip() if cur else part
            remaining = len(parts) - i - 1
            # start a new piece once this one is at its share, but never leave
            # fewer clauses than pieces still to fill
            if cur and secs(trial) > target and remaining >= (n - len(out) - 1):
                out.append(cur)
                cur = part
            else:
                cur = trial
        if cur:
            out.append(cur)
        return out

    def cut(text):
        if secs(text) <= max_secs:
            return [text]
        out = []
        # full stops first, then commas - both are places a pause belongs
        for sent in _SENT_SEAM.split(text):
            sent = sent.strip()
            if not sent:
                continue
            if secs(sent) <= max_secs:
                out.append(sent)
                continue
            pieces = []
            for part in _CLAUSE.split(sent):
                part = part.strip()
                if not part:
                    continue
                # over the cap but nothing to split on: leave it intact, unless
                # it is long enough that F5-TTS would split it for us
                if secs(part) > hard_secs:
                    pieces.extend(by_words(part))
                else:
                    pieces.append(part)
            out.extend(pack(pieces))
        return out or [text]

    def rejoin(pieces):
        """Glue anything too short back onto a neighbour.

        split_sentences merges sentences under MIN_SENT_CHARS so F5-TTS is
        never handed a tiny utterance, but cutting at full stops here undid
        that: "डॉ. अनन्या कुलकर्णी. एम बी बी एस." came back apart as three
        fragments, the first of them 0.4 seconds long.
        """
        out = []
        for p in pieces:
            if out and (len(out[-1]) < MIN_SENT_CHARS or len(p) < MIN_SENT_CHARS) \
                    and secs(out[-1] + " " + p) <= max_secs * 1.25:
                out[-1] = out[-1] + " " + p
            else:
                out.append(p)
        return out

    grown = []
    for text, kind in items:
        pieces = rejoin(cut(text))
        for i, p in enumerate(pieces):
            if i == len(pieces) - 1:
                grown.append((p, kind))
                continue
            # Only a sentence-ender is a stop. _CLAUSE splits AFTER the comma,
            # so every mid-sentence piece ends with one - and counting a comma
            # as "ends on punctuation" gave all of them the full 250ms stop,
            # which is what made a split sentence sound like two sentences.
            ends_sentence = re.search(r"[।\.\?\!…]\s*[\"'”’]?\s*$", p)
            grown.append((p, "chunk" if ends_sentence else "flow"))
    return grown


SHORT_PARA_CHARS = 140
_DIALOGUE_OPEN = ('"', '“', "'", '‘', '—', '-')


def split_blocks(text, max_chars=400, per_sentence=True,
                 short_para_chars=SHORT_PARA_CHARS):
    """Split while RESPECTING your line breaks.

    Collapsing all whitespace destroyed exactly the pauses you wrote into the
    script, which is why lines ran together. Each returned item is
    (chunk_text, pause_kind) where pause_kind says how much silence follows:
    'chunk' = mid-sentence, 'sent' = a full stop, 'line' = you pressed Enter,
    'spara' = a SHORT blank-line block, 'para' = a substantial one.

    'spara' exists because these scripts write dialogue one exchange per
    paragraph:

        आजी म्हणायची - "त्या झाडावर मुंज्या राहतो."
        <blank>
        "मुंज्या म्हणजे काय, आजी?"
        <blank>
        "एक मुलगा. तुझ्याच वयाचा."

    Treating those blank lines as scene changes put a 800ms dead stop between
    every line of a rapid back-and-forth. Measured across six stories, the
    proportion of blank-line breaks tracks perceived choppiness almost exactly
    (83 chars/paragraph in the worst, 174 in the best). A one-line reply is not
    a scene change and must not be read as one.
    """
    items = []
    para_start = 0

    def close_para():
        """Mark the break that just ended a paragraph, sized to the paragraph."""
        nonlocal para_start
        if not items or para_start >= len(items):
            return                              # consecutive blank lines
        ptext = " ".join(c for c, _ in items[para_start:]).strip()
        dialogue = ptext.startswith(_DIALOGUE_OPEN)
        items[-1][1] = "spara" if (len(ptext) <= short_para_chars or dialogue) else "para"
        para_start = len(items)

    for raw in (text or "").split("\n"):
        s = raw.strip()
        if not s:                                  # blank line -> longer pause
            close_para()
            continue
        if per_sentence:
            parts = split_sentences(s, max_chars)
        else:
            cs = split_text(s, max_chars)
            parts = [(c, "chunk") for c in cs]
        for i, (c, k) in enumerate(parts):
            items.append([c, k if i < len(parts) - 1 else "line"])
    if items:
        items[-1][1] = "end"
    return [(c, k) for c, k in items]


def estimate(script, max_chars, nfe, ref_wav):
    """Rough ETA so you know before you start whether to go make tea."""
    n = len(split_blocks(script, int(max_chars)))
    if n == 0:
        return "Nothing to generate yet."
    ref_s = 9.0
    try:
        info = sf.info(ref_wav)
        ref_s = info.duration
    except Exception:
        pass
    # measured on a GTX 1650: ~46 s per call at NFE 32 for ref 9 s + ~7 s speech
    per_call_ref = {"cuda": 1.9, "mps": 0.8, "cpu": 9.0}[DEVICE]
    secs_per_audio_sec = per_call_ref * (int(nfe) / 32.0)
    gen_s_per_chunk = int(max_chars) / 30.0
    total = n * (ref_s + gen_s_per_chunk) * secs_per_audio_sec
    mins = total / 60.0
    return (f"{n} chunk(s) · estimated ~{mins:.0f} min on {DEVICE.upper()} "
            f"(NFE {int(nfe)}, reference {ref_s:.1f}s). "
            f"Lower NFE and a shorter reference clip are the two big speed levers.")


# ------------------------------------------------------------ generation ---
def single_batch_limit(ref_wav, ref_text):
    """Largest chunk F5-TTS will still render in ONE pass.

    F5-TTS budgets roughly 22 s of audio per pass (reference + speech) and
    silently splits anything longer, cross-fading the halves itself. Those
    internal seams are where words get slurred and sentences sound broken.
    Staying under this limit means every join is ours, with a real pause.
    """
    try:
        dur = sf.info(ref_wav).duration
    except Exception:
        dur = 8.0
    ref_bytes = len(ref_text.encode("utf-8"))
    if dur <= 0 or ref_bytes == 0:
        return 200
    budget_bytes = ref_bytes / dur * max(1.0, 22.0 - dur)
    # Devanagari averages ~3 bytes/char; keep 10% headroom
    return max(80, int(budget_bytes / 3 * 0.9))


# Hard ceiling on how much more time than the estimate a chunk may be given.
# Above roughly 1.5x the model stops leaving the surplus as silence and starts
# generating confident nonsense to fill it. See plan_durations.
MAX_ROOMINESS = 1.45


def ref_profile(ref_wav, ref_text):
    """The reference exactly as F5-TTS will see it.

    F5-TTS does not use your wav as-is: preprocess_ref_audio_text strips the
    silent edges, clips anything over 12 s and appends 50 ms. Timing against
    the file on disk would therefore be timing against the wrong length. It
    caches on the file's md5, so asking for it here costs nothing later.
    """
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text
    path, text = preprocess_ref_audio_text(
        ref_wav, ref_text, show_info=lambda *a, **k: None)
    return path, text, sf.info(path).duration


def plan_durations(items, ref_text, ref_sec, speed=1.0, pace=1.0, log=print):
    """Seconds to allot each chunk, from syllables rather than bytes.

    Returns None if the pacing module is unavailable, in which case we fall
    back to F5-TTS's own byte estimate.
    """
    if pacing is None:
        return None
    rate = pacing.speech_rate(ref_text, ref_sec)
    log("  " + pacing.rate_report(ref_text, ref_sec))
    # The whole output inherits the reference clip's tempo. A brisk clip makes
    # every story brisk, no matter what the text is.
    if rate > 7.5:
        log(f"  NOTE: that reference is quick for narration "
            f"(~{rate:.1f} vs ~6 moras/sec). Everything will inherit its tempo - "
            f"raise Roominess to ~1.2, or record a calmer reference clip.")
    # speed is folded in here because fix_duration overrides F5-TTS's own
    # speed handling entirely - lower speed simply means more time.
    #
    # Ceiling is not arbitrary. Given a canvas far larger than the text needs,
    # the model does not leave the extra as silence - it fills it. Measured on
    # "मी म्हणालो, मी एकटाच आहे.": 1.4x estimate was clean, 1.8x came back as
    # fluent-sounding babble that an ASR transcribed as Kannada. Over-allotting
    # is not a safe direction to err in.
    roominess = min(MAX_ROOMINESS, 1.08 * float(pace) / max(0.4, float(speed)))
    headroom = max(2.0, 22.0 - ref_sec)
    out, clipped = [], 0
    for chunk, _ in items:
        est = pacing.estimate_seconds(chunk, rate, roominess=roominess)
        if est > headroom:
            est, clipped = headroom, clipped + 1
        out.append(est)
    if clipped:
        log(f"  {clipped} chunk(s) hit the {headroom:.1f}s per-pass ceiling - "
            f"use a shorter reference clip or smaller chunks")
    return out, rate


def synthesize(script, ckpt, ref_wav, ref_txt, speed, nfe, max_chars,
               pauses, use_dict, dict_rows, auto_translit, device,
               out_name=None, on_progress=None, log=print,
               cfg=2.0, sway=-1.0, trim=True, drop_directions=True,
               expand_nums=True, one_pass=True,
               pace=1.0, fit_duration=True, per_sentence=True,
               seed=None, retry_short=True, max_secs=3.2,
               lead_ms=350, tail_ms=900, seed_per_chunk=True,
               chain=True, chain_reanchor=8, drift_limit=0.045,
               chain_across_paragraphs=True,
               chain_anchored=True, apply_warmth=True):
    """Core generation. Shared by the UI and the overnight queue runner."""
    vocab = str(MODELS_DIR / "vocab.txt")
    if not Path(vocab).exists():
        raise RuntimeError(f"vocab.txt missing from {MODELS_DIR}")

    text = prepare_text(script, rows_to_dict(dict_rows), use_dict, auto_translit,
                        drop_directions, expand_nums)

    if one_pass:
        cap = single_batch_limit(ref_wav, ref_txt)
        if int(max_chars) > cap:
            log(f"  chunk size {int(max_chars)} -> {cap} so every chunk renders "
                f"in a single pass (avoids F5-TTS's internal seams)")
            max_chars = cap
    # abbreviation periods are hidden for the whole of chunking, then restored
    items = split_blocks(protect_abbrevs(text), int(max_chars),
                         per_sentence=per_sentence)
    if not items:
        raise RuntimeError("Nothing to say.")

    # Budget each chunk's duration ourselves. See pacing.py for why F5-TTS's
    # own byte-count estimate rushes some lines and drawls others.
    def _planning_failed(why):
        """Never let a fallback to the byte-count estimate be silent.

        Falling back reinstates the original pacing bug: the model races, and
        the only symptom is that the audio sounds rushed. Callers that suppress
        `log` - every A/B harness in tools/ did - would otherwise get hurried
        audio with no indication why. Both sides of the paragraph A/B rendered
        with planned_s=0 and were judged rushed, which silently invalidated the
        comparison. So this goes to stderr regardless of the log callback.
        """
        import traceback
        print(f"\n*** DURATION PLANNING FAILED: {why}\n"
              f"*** Falling back to F5-TTS's byte-count estimate.\n"
              f"*** THE OUTPUT WILL SOUND RUSHED - do not judge quality from it.\n"
              f"{traceback.format_exc() if sys.exc_info()[0] else ''}",
              file=sys.stderr, flush=True)

    ref_norm, ref_sec, plan, rate = ref_txt.strip(), 0.0, None, 0.0
    if fit_duration:
        try:
            _, ref_norm, ref_sec = ref_profile(ref_wav, ref_txt.strip())
            rate = pacing.speech_rate(ref_norm, ref_sec) if pacing else 0.0
            if rate and max_secs and max_secs > 0:
                before = len(items)
                items = split_by_duration(items, rate, float(max_secs),
                                          1.08 * float(pace) / max(0.4, float(speed)))
                if len(items) != before:
                    log(f"  {before} -> {len(items)} chunks, so none runs over "
                        f"{float(max_secs):.1f}s (long chunks slur ळ, मी and च)")
            planned = plan_durations(items, ref_norm, ref_sec,
                                     speed=speed, pace=pace, log=log)
            if planned:
                plan, rate = planned
            else:
                _planning_failed("plan_durations returned nothing")
        except Exception as e:
            _planning_failed(repr(e))
            log(f"  duration planning unavailable ({e}); using F5-TTS's estimate")
    if fit_duration and not plan:
        _planning_failed("no per-chunk budget was produced")

    # put the abbreviation periods back before anything is spoken or indexed -
    # unconditionally, so a failure above cannot leak the marker into the model
    items = [(restore_abbrevs(c), k) for c, k in items]

    tts = get_tts(ckpt, vocab, device)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = re.sub(r"[^A-Za-z0-9_\-]+", "_", out_name or "")[:40]
    run_dir = PARTS_DIR / (f"{tag}_{stamp}" if tag else stamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Index of what each chunk wav says. Without this, "chunk 47 sounds wrong"
    # cannot be traced back to the text that produced it, which is exactly the
    # lookup every QA pass needs.
    try:
        (run_dir / "chunks.tsv").write_text(
            "file\tkind\tplanned_s\ttext\n" + "\n".join(
                f"{i:04d}.wav\t{k}\t{(plan[i-1] if plan else 0):.2f}\t{c}"
                for i, (c, k) in enumerate(items, 1)),
            encoding="utf-8")
    except Exception:
        pass

    # --- prosody chaining ---------------------------------------------------
    # F5-TTS builds `ref_text + gen_text` as ONE utterance and generates the
    # whole thing conditioned on ref_audio, then slices the reference off. The
    # output is therefore a CONTINUATION of whatever audio it was given.
    #
    # Handing it the same 5s clip for every chunk is what makes a story sound
    # like 200 separate recordings: each chunk restarts the intonation from
    # that clip. Feeding it the PREVIOUS chunk instead lets the contour carry
    # across the join, which is the whole difference between a read-aloud list
    # of sentences and continuous narration.
    #
    # Drift is the risk - each chunk conditions on synthetic audio, so error
    # compounds. Re-anchoring to the real reference every `chain_reanchor`
    # chunks, and at every paragraph, bounds it.
    chain_dir = run_dir / "_chain"
    if chain:
        chain_dir.mkdir(exist_ok=True)
    prev_wav, prev_text, since_anchor = None, None, 0

    # Drift guard, calibrated by measurement rather than assumption.
    #
    # Chaining does NOT compound error the way it looks like it should: across
    # a 213-chunk story the chained run wandered +0.004 from the reference and
    # an unchained run of the same story wandered +0.005. Re-anchoring already
    # stops the runaway.
    #
    # What it does do is sit consistently further out - mean distance 0.015
    # chained against 0.013 unchained, worst 0.049 against 0.041. Every chunk
    # conditions on synthetic audio that is already slightly imperfect and
    # inherits that offset. Heard, that is a voice which is subtly not yours
    # throughout, rather than one that slides away.
    #
    # So: chain a short distance only (2), and re-anchor early on any chunk
    # that lands beyond 0.045 - just above the 0.041 that unchained generation
    # reaches naturally, so normal variation does not trip it.
    def _fingerprint(x, sr_):
        try:
            import librosa
            a = np.asarray(x, dtype=np.float32)
            if a.ndim > 1:
                a = a.mean(1)
            if a.size < sr_ // 4:
                return None
            mf = librosa.feature.mfcc(y=a, sr=sr_, n_mfcc=20)
            return np.concatenate([mf.mean(1), mf.std(1)])
        except Exception:
            return None

    def _cos(a, b):
        a, b = a - a.mean(), b - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return 1.0 - float(a @ b / d) if d else 1.0

    ref_fp = None
    if chain and drift_limit > 0:
        try:
            _rx, _rsr = sf.read(ref_wav)
            ref_fp = _fingerprint(_rx, _rsr)
        except Exception:
            ref_fp = None
    drifted = 0
    chained_n = 0            # durable evidence chaining actually engaged

    def reference_for(idx, kind_prev):
        """(path, text, seconds) to condition this chunk on."""
        nonlocal since_anchor, chained_n
        # A paragraph break used to force a re-anchor here. It must not.
        #
        # Pause length and prosody reset are unrelated concerns that were
        # conflated: a narrator pauses at a paragraph, they do not reset their
        # voice. Because these scripts are written one short paragraph per
        # line - dialogue especially - that rule fired constantly and chaining
        # only engaged on 45-70% of chunks, tracking perceived quality almost
        # exactly (the story rated best chained 65%, the two complained about
        # chained 45%). The `since_anchor >= chain_reanchor` counter it was
        # meant to back up almost never fired at all: 0-5 times in a whole
        # story. Drift is bounded by the anchor being genuine audio, not by
        # resetting at punctuation - see the +0.004 vs +0.005 measurement.
        if (not chain or prev_wav is None or since_anchor >= chain_reanchor
                or (not chain_across_paragraphs
                    and kind_prev in ("para", "spara", "line"))):
            since_anchor = 0
            return ref_wav, ref_txt.strip(), ref_sec
        p = chain_dir / f"prev_{idx:04d}.wav"

        # ANCHORED chaining: the real reference clip, then the previous chunk.
        #
        # Replacing the reference outright is what cost voice fidelity - the
        # model's only evidence of the speaker became synthetic audio that was
        # already slightly off, and every chunk inherited that (measured: mean
        # distance 0.015 chained against 0.013 unchained). Concatenating keeps
        # genuine human audio in the conditioning window, so timbre is anchored
        # to a real recording while the previous chunk still supplies the
        # prosodic run-up.
        #
        # The real clip goes FIRST on purpose. F5-TTS clips a reference over
        # 12s by accumulating from the start, so if anything is lost it is the
        # synthetic tail, never the human anchor.
        if chain_anchored:
            room = 11.0 - ref_sec               # stay clear of the 12s clip
            if prev_wav.size / sr <= room:
                try:
                    anchor, _ = sf.read(ref_wav, dtype="float32")
                    if anchor.ndim > 1:
                        anchor = anchor.mean(1)
                    gap = np.zeros(int(sr * 0.12), dtype=np.float32)
                    sf.write(p, np.concatenate([anchor, gap, prev_wav]), sr)
                    joined = ref_txt.strip().rstrip(".") + ". " + prev_text
                    _, norm_t, secs_ = ref_profile(str(p), joined)
                    since_anchor += 1
                    chained_n += 1
                    return str(p), norm_t, secs_
                except Exception:
                    pass
            since_anchor = 0                    # would not fit - anchor instead
            return ref_wav, ref_txt.strip(), ref_sec

        sf.write(p, prev_wav, sr)
        try:
            _, norm_t, secs_ = ref_profile(str(p), prev_text)
        except Exception:
            since_anchor = 0
            return ref_wav, ref_txt.strip(), ref_sec
        since_anchor += 1
        chained_n += 1
        return str(p), norm_t, secs_

    pieces, sr, t0 = [], 24000, time.time()
    prev_kind = None
    for i, (chunk, kind) in enumerate(items, 1):
        done = i - 1
        if on_progress:
            eta = ""
            if done:
                per = (time.time() - t0) / done
                eta = f" · ~{per*(len(items)-done)/60:.1f} min left"
            on_progress(done / len(items), f"Chunk {i}/{len(items)}{eta}")

        cur_ref, cur_ref_txt, cur_ref_sec = reference_for(i, prev_kind)
        kw = dict(ref_file=cur_ref, ref_text=cur_ref_txt.strip(),
                  speed=float(speed), nfe_step=int(nfe),
                  cfg_strength=float(cfg), sway_sampling_coef=float(sway),
                  remove_silence=False)   # we trim ourselves, see trim_silence
        if seed is not None:
            # seed_per_chunk offsets by the chunk number, so a run is
            # reproducible while each chunk still starts from different noise.
            # Turning it off uses the SAME seed for every chunk - which is what
            # the probes did, and those came out consistently clean. If one
            # seed happens to suit this voice and reference clip, there is no
            # reason to keep rolling for a worse one.
            kw["seed"] = int(seed) + i if seed_per_chunk else int(seed)
        want = plan[i - 1] if plan else None
        if want is not None:
            # fix_duration is the TOTAL canvas: reference + speech, and the
            # reference changes per chunk when chaining
            kw["fix_duration"] = cur_ref_sec + want
        try:
            wav, sr, _ = tts.infer(gen_text=chunk, **kw)
        except torch.cuda.OutOfMemoryError:
            # 4 GB cards run out on long chunks; halve it and carry on
            torch.cuda.empty_cache()
            log(f"  chunk {i}: out of VRAM, retrying in two halves")
            mid = len(chunk) // 2
            sp = chunk.rfind(" ", 0, mid) or mid
            halves = [h for h in (chunk[:sp].strip(), chunk[sp:].strip()) if h]
            wav = np.concatenate([
                np.asarray(tts.infer(gen_text=h, **kw)[0], dtype=np.float32)
                for h in halves])

        wav = np.asarray(wav, dtype=np.float32)
        if np.isnan(wav).any():
            raise RuntimeError(f"Chunk {i} produced NaN (float16 failure mode); "
                               f"switch device to cpu.")

        # Still at full voice at either edge = cut off mid-word. Give it more
        # room and render again; measured at 5% of chunks before this.
        clipped_head = starts_mid_word(wav, sr)
        if want is not None and retry_short and (ends_mid_word(wav, sr) or clipped_head):
            where = "started" if clipped_head else "ended"
            log(f"  chunk {i}: {where} mid-word, re-rendering with more time")
            try:
                kw3 = dict(kw)
                # `want` already carries the roominess; stacking another 30%
                # on top of a high pace setting lands in the babble zone, so
                # the extra is small and the total stays under the ceiling
                kw3["fix_duration"] = cur_ref_sec + want * 1.12
                w3, sr, _ = tts.infer(gen_text=chunk, **kw3)
                w3 = np.asarray(w3, dtype=np.float32)
                if (not np.isnan(w3).any() and not ends_mid_word(w3, sr)
                        and not starts_mid_word(w3, sr)):
                    wav = w3
            except Exception as e:
                log(f"  chunk {i}: re-render failed ({e}), keeping the first take")

        if trim:
            wav = trim_silence(wav, sr)

        # If the model used far less time than the text needs, it very likely
        # skipped something - this is the "whole word missing" failure. One
        # re-roll on a different seed usually lands it, and costs one pass.
        # (a short line legitimately finishes early, so only judge longer ones)
        if (want is not None and retry_short and want > 2.0
                and len(wav) / sr < want * 0.58):
            log(f"  chunk {i}: {len(wav)/sr:.1f}s used of {want:.1f}s planned "
                f"- suspect a dropped word, re-rolling once")
            try:
                kw2 = dict(kw)
                kw2["seed"] = (int(seed) + i if seed is not None
                               else random.randint(0, 2**31 - 1))
                kw2["fix_duration"] = cur_ref_sec + want * 1.06
                w2, sr, _ = tts.infer(gen_text=chunk, **kw2)
                w2 = np.asarray(w2, dtype=np.float32)
                if not np.isnan(w2).any():
                    if trim:
                        w2 = trim_silence(w2, sr)
                    if len(w2) > len(wav):      # keep whichever said more
                        wav = w2
            except Exception as e:
                log(f"  chunk {i}: re-roll failed ({e}), keeping the first take")

        wav = fade_edges(wav, sr)         # no clicks where chunks meet

        sf.write(run_dir / f"{i:04d}.wav", wav, sr)   # never lose a long run
        if chain:
            # only chain from a take we believe in - conditioning the next
            # chunk on a bad one propagates the damage
            healthy = (wav.size > int(sr * 0.3) and not np.isnan(wav).any()
                       and float(np.abs(wav).max()) > 1e-3)
            if healthy and ref_fp is not None:
                fp = _fingerprint(wav, sr)
                if fp is not None and _cos(ref_fp, fp) > drift_limit:
                    healthy = False          # voice has wandered - go back
                    drifted += 1
            prev_wav, prev_text = (wav, chunk) if healthy else (None, None)
        prev_kind = kind
        if chain and i % 25 == 0:
            # preprocess_ref_audio_text caches by md5 and never evicts; with a
            # fresh reference per chunk that grows without bound
            try:
                from f5_tts.infer import utils_infer as _ui
                _ui._ref_audio_cache.clear()
                _ui._ref_text_cache.clear()
            except Exception:
                pass
        pieces.append(wav)
        if device == "cuda":
            torch.cuda.empty_cache()

        gap = pauses.get(kind, 0)
        if gap > 0 and i < len(items):
            pieces.append(np.zeros(int(sr * gap / 1000), dtype=np.float32))

    if chain and drifted:
        log(f"  re-anchored {drifted} time(s) on drift "
            f"(voice had wandered past {drift_limit:.3f} from the reference)")

    audio = np.concatenate(pieces)

    # Breathing room at the very edges. trim_silence deliberately strips each
    # chunk's own head and tail so the pauses between them are exact - but that
    # leaves the FINISHED file starting 125ms in and, measured across the
    # overnight run, ending 0-160ms after the last syllable. Several stories
    # stopped dead at 0ms. That is what an abrupt start and a chopped ending
    # actually are: not a truncated word, just no room around the speech.
    audio = np.concatenate([
        np.zeros(int(sr * lead_ms / 1000), dtype=np.float32),
        audio,
        np.zeros(int(sr * tail_ms / 1000), dtype=np.float32)])

    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = audio / peak * 0.95
    base = f"{tag}_{stamp}" if tag else f"story_{stamp}"
    out_path = OUT_DIR / f"{base}.wav"
    if apply_warmth and warmth is not None:
        # Vocos decodes accurate speech that is also dry and thin. The raw
        # version is kept beside it so this stays reversible - and so an A/B
        # is always one file away.
        try:
            sf.write(OUT_DIR / f"{base}_raw.wav", audio, sr)
            audio = warmth.process(audio, sr)
            log("  warmth applied (raw kept as _raw.wav)")
        except Exception as e:
            log(f"  warmth skipped ({e})")
    sf.write(out_path, audio, sr)

    # A durable record of what this render ACTUALLY used. _chain/ is deleted
    # below, so without this there is no way afterwards to tell whether
    # chaining engaged, whether duration budgeting applied, or which reference
    # produced a given wav - and this session lost hours to renders that were
    # silently degraded in exactly those ways.
    try:
        planned_ok = bool(plan) and all(x > 0 for x in plan)
        (run_dir / "run_info.txt").write_text(
            f"output        {out_path.name}\n"
            f"model         {Path(ckpt).name}\n"
            f"reference     {Path(ref_wav).name}  "
            f"({ref_sec:.2f}s, {rate:.2f} moras/sec)\n"
            f"chunks        {len(items)}\n"
            f"CHAINED       {chained_n} of {len(items)} "
            f"({chained_n * 100 // max(len(items), 1)}%)"
            f"{'' if chain else '   [chaining OFF]'}\n"
            f"  reanchor    every {chain_reanchor}, anchored={chain_anchored}, "
            f"across_paragraphs={chain_across_paragraphs}, drift_limit={drift_limit}\n"
            f"  drifted     {drifted}\n"
            f"DURATION      "
            f"{'budgeted per chunk' if planned_ok else '*** UNPACED - byte-count fallback ***'}\n"
            f"nfe           {nfe}\n"
            f"cfg           {cfg}\n"
            f"roominess     {pace}\n"
            f"max_secs      {max_secs}\n"
            f"seed          {seed}  (per_chunk={seed_per_chunk})\n"
            f"pauses        {pauses}\n"
            f"warmth        {bool(apply_warmth and warmth)}\n"
            f"lead/tail ms  {lead_ms}/{tail_ms}\n",
            encoding="utf-8")
    except Exception:
        pass
    if chain:
        log(f"  chained {chained_n} of {len(items)} chunks "
            f"({chained_n * 100 // max(len(items), 1)}%)")
        shutil.rmtree(run_dir / "_chain", ignore_errors=True)
    return out_path, len(audio) / sr, time.time() - t0, len(items), run_dir


def generate(script, ckpt, ref_wav, ref_txt, speed, nfe, pause_ms, max_chars,
             use_dict, dict_rows, auto_translit, device_choice,
             line_pause, para_pause, title, cfg, trim, drop_dir,
             expand_nums, one_pass, sent_pause, pace, fit_dur, per_sent,
             max_secs, lead_ms, tail_ms, flow_pause,
             spara_pause=320, chain=True, chain_across_para=True,
             chain_reanchor=8, drift_limit=0.045, apply_warmth=True,
             seed_num=7, lock_seed=True,
             progress=gr.Progress()):
    if not (script or "").strip():
        return None, "Type or paste some Marathi text first."
    if not ckpt:
        return None, f"No checkpoint in {MODELS_DIR}"
    if not ref_wav:
        return None, f"No reference clip in {REF_DIR}"
    if not (ref_txt or "").strip():
        return None, "Reference transcript is empty - it must match the clip word for word."

    device = DEVICE if device_choice == "auto" else device_choice
    pauses = {"chunk": int(pause_ms), "flow": int(flow_pause), "sent": int(sent_pause),
              "line": int(line_pause), "spara": int(spara_pause),
              "para": int(para_pause), "end": 0}
    progress(0, desc=f"Loading model on {device}...")
    try:
        out_path, dur, took, n, run_dir = synthesize(
            script, ckpt, ref_wav, ref_txt, speed, nfe, max_chars, pauses,
            use_dict, dict_rows, auto_translit, device, out_name=title,
            on_progress=lambda f, d: progress(f, desc=d),
            cfg=cfg, trim=trim, drop_directions=drop_dir,
            expand_nums=expand_nums, one_pass=one_pass,
            pace=pace, fit_duration=fit_dur, per_sentence=per_sent,
            max_secs=max_secs, lead_ms=lead_ms, tail_ms=tail_ms,
            seed=int(seed_num) if seed_num is not None else None,
            seed_per_chunk=not lock_seed,
            chain=chain, chain_reanchor=int(chain_reanchor),
            chain_across_paragraphs=chain_across_para,
            drift_limit=float(drift_limit), apply_warmth=apply_warmth)
    except Exception as e:
        return None, f"Failed: {e}"

    return str(out_path), (
        f"Done · {dur/60:.1f} min of audio from {n} chunks in {took/60:.1f} min "
        f"({dur/max(took,1e-6):.2f}x realtime) on {device}\n"
        f"Saved: {out_path}\nPer-chunk parts: {run_dir}"
    )


# --------------------------------------------------------- reference mgmt --
# ------------------------------------------------------------- job queue ---
QUEUE_DIR = BASE / "queue"
QUEUE_DONE = QUEUE_DIR / "done"
QUEUE_FAILED = QUEUE_DIR / "failed"
for d in (QUEUE_DIR, QUEUE_DONE, QUEUE_FAILED):
    d.mkdir(parents=True, exist_ok=True)


def list_queue():
    return sorted(p.name for p in QUEUE_DIR.glob("*.txt"))


def queue_table():
    rows = []
    for n in list_queue():
        p = QUEUE_DIR / n
        words = len(p.read_text(encoding="utf-8", errors="ignore").split())
        rows.append([n, f"{words:,} words"])
    return rows or [["(queue is empty)", ""]]


def add_to_queue(title, script):
    if not (script or "").strip():
        return queue_table(), "Nothing to add - the script box is empty."
    tag = re.sub(r"[^A-Za-z0-9_\-]+", "_", (title or "").strip()) or "story"
    idx = len(list(QUEUE_DIR.glob("*.txt"))) + 1
    path = QUEUE_DIR / f"{idx:02d}_{tag}.txt"
    path.write_text(script, encoding="utf-8")
    return queue_table(), f"Queued: {path.name}"


def clear_queue():
    for p in QUEUE_DIR.glob("*.txt"):
        p.unlink()
    return queue_table(), "Queue cleared (finished files in done/ are untouched)."


def run_queue(ckpt, ref_wav, ref_txt, speed, nfe, pause_ms, max_chars,
              use_dict, dict_rows, auto_translit, device_choice,
              line_pause, para_pause, cfg=2.0, trim=True, drop_dir=True,
              expand_nums=True, one_pass=True, sent_pause=260, pace=1.0,
              fit_dur=True, per_sent=True, max_secs=3.2,
              lead_ms=350, tail_ms=900, flow_pause=60,
              spara_pause=320, chain=True, chain_across_para=True,
              chain_reanchor=8, drift_limit=0.045, apply_warmth=True,
              seed_num=7, lock_seed=True,
              progress=gr.Progress()):
    """Process every queued story. Designed to be left running overnight:
    one bad story never stops the rest, and finished work is never redone."""
    jobs = list_queue()
    if not jobs:
        return queue_table(), "Queue is empty."
    if not ckpt or not ref_wav or not (ref_txt or "").strip():
        return queue_table(), "Pick a checkpoint, a reference clip and its transcript first."

    device = DEVICE if device_choice == "auto" else device_choice
    pauses = {"chunk": int(pause_ms), "flow": int(flow_pause), "sent": int(sent_pause),
              "line": int(line_pause), "spara": int(spara_pause),
              "para": int(para_pause), "end": 0}
    log_lines, t_all = [], time.time()

    for j, name in enumerate(jobs, 1):
        src = QUEUE_DIR / name
        if not src.exists():
            continue
        title = src.stem
        progress((j - 1) / len(jobs), desc=f"Story {j}/{len(jobs)}: {title}")
        log_lines.append(f"[{j}/{len(jobs)}] {title} ...")
        try:
            script = src.read_text(encoding="utf-8")
            out_path, dur, took, n, _rd = synthesize(
                script, ckpt, ref_wav, ref_txt, speed, nfe, max_chars, pauses,
                use_dict, dict_rows, auto_translit, device, out_name=title,
                on_progress=lambda f, d, _j=j: progress(
                    ((_j - 1) + f) / len(jobs), desc=f"{title}: {d}"),
                log=lambda m: log_lines.append("   " + m),
                cfg=cfg, trim=trim, drop_directions=drop_dir,
                expand_nums=expand_nums, one_pass=one_pass,
                pace=pace, fit_duration=fit_dur, per_sentence=per_sent,
            max_secs=max_secs, lead_ms=lead_ms, tail_ms=tail_ms,
            seed=int(seed_num) if seed_num is not None else None,
            seed_per_chunk=not lock_seed,
            chain=chain, chain_reanchor=int(chain_reanchor),
            chain_across_paragraphs=chain_across_para,
            drift_limit=float(drift_limit), apply_warmth=apply_warmth)
            shutil.move(str(src), str(QUEUE_DONE / name))
            log_lines.append(f"   OK  {dur/60:.1f} min audio, {n} chunks, "
                             f"{took/60:.1f} min -> {out_path.name}")
        except Exception as e:                      # keep the night running
            shutil.move(str(src), str(QUEUE_FAILED / name))
            log_lines.append(f"   FAILED: {e}")

    total = (time.time() - t_all) / 60
    log_lines.append(f"\nQueue finished in {total:.1f} min. "
                     f"Audio in {OUT_DIR}, sources moved to done/ or failed/.")
    return queue_table(), "\n".join(log_lines)


def save_reference(upload_path, name, transcript):
    if not upload_path:
        return "Choose or record a wav first.", gr.update(), gr.update()
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", (name or "").strip()) or f"ref_{int(time.time())}"
    if not (transcript or "").strip():
        return ("Add the transcript - the exact words spoken in this clip. "
                "This matters more than almost anything else for quality."), gr.update(), gr.update()

    dst = REF_DIR / f"{name}.wav"
    try:
        x, sr = sf.read(upload_path, dtype="float32", always_2d=True)
        x = x.mean(1) if x.shape[1] > 1 else x[:, 0]
        if sr != 24000:                       # model works at 24 kHz mono
            import soxr
            x = soxr.resample(x, sr, 24000, quality="VHQ")
            sr = 24000
        p = float(np.abs(x).max())
        if p > 0:
            x = x / p * 0.95
        sf.write(dst, x, sr, subtype="PCM_16")
    except Exception as e:
        return f"Could not read that audio: {e}", gr.update(), gr.update()

    dst.with_suffix(".txt").write_text(transcript.strip(), encoding="utf-8")
    dur = len(x) / sr
    warn = ""
    if dur > 12:
        warn = "  ⚠ over 12 s - F5-TTS will clip it; 6-10 s works best."
    elif dur < 3:
        warn = "  ⚠ under 3 s - quite short for a stable voice match."
    refs = list_refs()
    return (f"Saved {dst.name} ({dur:.1f}s).{warn}",
            gr.update(choices=refs, value=str(dst)), transcript.strip())


def autotranscribe(upload_path):
    if not upload_path:
        return "Choose a wav first."
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return "faster-whisper is not installed in this environment."
    m = WhisperModel("large-v3", device="cpu", compute_type="int8")
    segs, _ = m.transcribe(upload_path, language="mr", beam_size=5)
    return " ".join(s.text.strip() for s in segs).strip()


# ------------------------------------------------------------------- UI ----
STUDIO_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#0b0d10",
    background_fill_primary="#13171d",
    background_fill_secondary="#0f1318",
    block_background_fill="#13171d",
    block_border_width="1px",
    block_border_color="#232a33",
    block_radius="14px",
    block_label_background_fill="transparent",
    block_label_text_color="#8b97a8",
    block_title_text_color="#c8d2e0",
    body_text_color="#dfe6ef",
    body_text_color_subdued="#8b97a8",
    input_background_fill="#0d1116",
    input_border_color="#242b35",
    input_radius="10px",
    button_large_radius="10px",
    button_small_radius="8px",
    button_primary_background_fill="linear-gradient(180deg,#f0a52b,#d98613)",
    button_primary_background_fill_hover="linear-gradient(180deg,#ffb63c,#e8951c)",
    button_primary_text_color="#1a1204",
    button_secondary_background_fill="#1b212a",
    button_secondary_background_fill_hover="#232b36",
    button_secondary_text_color="#dfe6ef",
    slider_color="#f0a52b",
)

STUDIO_CSS = """
.gradio-container{max-width:1500px!important;padding-top:0!important}
footer{display:none!important}

#studio-head{display:flex;align-items:center;gap:16px;flex-wrap:wrap;
  padding:14px 20px;margin:0 0 14px;border:1px solid #232a33;border-radius:14px;
  background:linear-gradient(135deg,#151a21 0%,#11151a 60%,#171208 100%)}
#studio-head .brand{display:flex;align-items:baseline;gap:10px}
#studio-head .brand b{font-size:19px;letter-spacing:-.02em;color:#f3f6fa}
#studio-head .brand span{font-size:12px;color:#7d8898}
#studio-head .pills{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap}
.pill{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;
  font-weight:600;letter-spacing:.02em;padding:5px 11px;border-radius:999px;
  border:1px solid #2a323d;background:#161b22;color:#9fadbf;white-space:nowrap}
.pill.on{border-color:#3c2f12;background:#20180a;color:#f0a52b}
.pill .dot{width:6px;height:6px;border-radius:50%;background:currentColor}

.tabs>.tab-nav{gap:2px;border-bottom:1px solid #232a33!important;padding-bottom:0}
.tabs>.tab-nav>button{border:0!important;background:transparent!important;
  color:#7d8898!important;font-weight:600;font-size:13.5px;padding:10px 16px!important;
  border-radius:9px 9px 0 0!important}
.tabs>.tab-nav>button.selected{color:#f0a52b!important;background:#13171d!important;
  box-shadow:inset 0 -2px 0 #f0a52b}

.block{box-shadow:none!important}
span[data-testid="block-info"]{color:#6f7b8c!important;font-size:11.5px!important;
  line-height:1.5!important}
label>span{font-size:12.5px!important;font-weight:600!important;letter-spacing:.01em}

.gradio-accordion{border:1px solid #232a33!important;border-radius:12px!important;
  background:#10141a!important;overflow:hidden}
.gradio-accordion>button,.gradio-accordion .label-wrap{font-weight:650!important;
  font-size:13px!important;color:#c8d2e0!important;letter-spacing:.01em}

#script-box textarea{font-size:15.5px!important;line-height:1.75!important;
  background:#0a0e12!important;border-color:#242b35!important}
#script-box textarea::placeholder{color:#4c5665}

#status-box textarea,#est-box textarea,#qlog textarea{
  font-family:var(--font-mono)!important;font-size:12px!important;
  line-height:1.6!important;color:#a8b6c8!important;background:#0a0e12!important}

#go-btn{font-weight:700!important;font-size:14.5px!important;letter-spacing:.01em;
  box-shadow:0 6px 18px -8px rgba(240,165,43,.6)!important}

table tbody td{font-size:12.5px!important;color:#c2cdda!important}
table thead th{font-size:11.5px!important;text-transform:uppercase;
  letter-spacing:.06em;color:#7d8898!important}
"""


def _pill(text, on=False):
    return f'<span class="pill{" on" if on else ""}"><i class="dot"></i>{text}</span>'


# Gradio 6 moved theme/css from the Blocks constructor to launch().
with gr.Blocks(title="Marathi Story Voice") as demo:
    gr.HTML(
        '<div id="studio-head">'
        '<div class="brand"><b>🎙️ Marathi Story Voice</b>'
        '<span>भयकथा narration studio</span></div>'
        '<div class="pills">'
        + _pill(device_label(), True)
        + _pill("float32")
        + _pill(f"{len(list_ckpts())} models")
        + _pill(f"{len(list_refs())} reference clips")
        + _pill("chaining + warmth", True)
        + "</div></div>")

    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column(scale=3):
                script = gr.Textbox(label="Script (Marathi)", lines=20,
                                    elem_id="script-box",
                                    placeholder="इथे तुमची भयकथा पेस्ट करा…")
                with gr.Row():
                    go = gr.Button("Generate audio", variant="primary", scale=2,
                                   elem_id="go-btn")
                    est_btn = gr.Button("Estimate time", scale=1)
                est = gr.Textbox(label="Estimate", lines=2, interactive=False,
                                 elem_id="est-box")
                audio_out = gr.Audio(
                    label="Result", type="filepath",
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#3d4757",
                        waveform_progress_color="#f0a52b",
                        show_recording_waveform=True))
                status = gr.Textbox(label="Status", lines=6, interactive=False,
                                    elem_id="status-box")

            with gr.Column(scale=2):
                ckpt = gr.Dropdown(list_ckpts(), label="Checkpoint",
                                   value=(list_ckpts() or [None])[0])
                ref_wav = gr.Dropdown(list_refs(), label="Reference clip",
                                      value=(list_refs() or [None])[0])
                ref_txt = gr.Textbox(label="Reference transcript (must match exactly)",
                                     lines=3, value=ref_text_for((list_refs() or [""])[0]))
                refresh = gr.Button("↻ Refresh lists", size="sm")
                title = gr.Textbox(label="Title (used for the output filename)",
                                   placeholder="munjya")
                with gr.Accordion("Speed / quality", open=True):
                    mood = gr.Dropdown(
                        ["custom", "whisper / creepy (0.80)", "atmospheric (0.85)",
                         "normal narration (1.00)", "tense / urgent (1.10)",
                         "chase / panic (1.15)"],
                        value="normal narration (1.00)", label="Delivery preset")
                    speed = gr.Slider(0.5, 1.5, 1.0, step=0.05, label="Speed")
                    nfe = gr.Slider(8, 64, 16, step=2,
                                    label="NFE steps (16 draft · 32 final)",
                                    info="32 is the sweet spot. Above ~40 the solver "
                                         "over-sharpens and adds the buzzy artifacts - "
                                         "more steps is not more quality here.")
                    cfg = gr.Slider(1.0, 3.0, 2.0, step=0.1,
                                    label="CFG strength (1.5 loose · 2.0 balanced · "
                                          "2.5 tight · 3.0 can sound stiff)")
                    max_chars = gr.Slider(150, 700, 400, step=50,
                                          label="Chunk size (bigger = fewer, faster passes)")
                    device_choice = gr.Radio(["auto", "cuda", "cpu"], value="auto",
                                             label="Device")
                with gr.Accordion("Script handling", open=True):
                    drop_dir = gr.Checkbox(
                        True, label="Skip stage directions",
                        info="Drops [ ... ] and *[ ... ]* cues instead of reading "
                             "them aloud.")
                    trim = gr.Checkbox(
                        True, label="Trim silence around each chunk",
                        info="Removes the model's ragged head/tail silence so the "
                             "pause settings below are exact. Gentle by design - "
                             "quiet trailing consonants are preserved.")
                    expand_nums = gr.Checkbox(
                        True, label="Read numbers as Marathi words",
                        info="३०४ → तीनशे चार · 1994 → एकोणीसशे चौऱ्याण्णव · "
                             "2019 → दोन हजार एकोणीस · 09 → शून्य नऊ")
                    one_pass = gr.Checkbox(
                        True, label="Force single-pass chunks (recommended)",
                        info="Caps chunk size so F5-TTS never splits a chunk "
                             "internally. Its hidden cross-fades are what make "
                             "sentences sound broken or words slurred.")
                    per_sent = gr.Checkbox(
                        True, label="One sentence per chunk (recommended)",
                        info="Gives every sentence its own start, end and pause "
                             "instead of letting the model run them together.")
                with gr.Accordion("Pacing", open=True):
                    gr.Markdown(
                        "F5-TTS decides how long a line may take from its "
                        "**UTF-8 byte count**. In Devanagari that is close to "
                        "meaningless — a matra or halant costs 3 bytes and adds "
                        "no time, so `स्वप्न` is given 2.3× the time per syllable "
                        "that `कमल` gets. Starved lines rush, race, run "
                        "sentences together, clip their last word, and in the "
                        "worst case **drop a word entirely**.\n\n"
                        "With this on the budget is computed from actual "
                        "syllables, calibrated against your reference clip's own "
                        "measured speaking rate."
                    )
                    fit_dur = gr.Checkbox(
                        True, label="Budget duration by syllables (recommended)",
                        info="Off = F5-TTS's byte-count guess, i.e. the old "
                             "behaviour.")
                    max_secs = gr.Slider(
                        0.0, 8.0, 3.2, step=0.2,
                        label="Max seconds per chunk",
                        info="The model pronounces ळ, मी and च correctly in "
                             "short phrases and slurs them in long ones. "
                             "Measured: probe phrases 1-2s were all correct, "
                             "story chunks ran 4s median / 15s worst and were "
                             "not. 0 disables the cap.")
                    pace = gr.Slider(
                        0.85, 1.35, 1.35, step=0.01, label="Roominess",
                        info="Extra time on top of the estimate. 1.35 is the "
                             "tested default: the same sentence on 10 random "
                             "seeds was clean 4 times out of 10 at 1.0, and 9 "
                             "out of 10 here. Starved chunks rush whichever "
                             "word is hardest, and which word that is changes "
                             "with the seed. Do not go higher - the ceiling "
                             "exists because a too-large canvas gets filled "
                             "with invented speech.")
                with gr.Accordion("Pauses", open=True):
                    gr.Markdown(
                        "Your **line breaks are respected**: press Enter for a "
                        "beat, leave a blank line for a longer one. Earlier "
                        "versions collapsed them, which is why lines ran together."
                    )
                    pause_ms = gr.Slider(0, 800, 150, step=25,
                                         label="Within a long line (ms)")
                    flow_pause = gr.Slider(0, 400, 60, step=10,
                                           label="Inside a split sentence (ms)",
                                           info="When one sentence is too long "
                                                "and gets divided, this is the "
                                                "gap between its pieces. Small "
                                                "on purpose - it should read as "
                                                "a breath, not a full stop.")
                    sent_pause = gr.Slider(0, 1200, 260, step=20,
                                           label="At a full stop (ms)",
                                           info="Only applies with one-sentence-"
                                                "per-chunk on. Inside a chunk the "
                                                "model chooses how long to rest at "
                                                "a '.', and under time pressure it "
                                                "chooses not to.")
                    line_pause = gr.Slider(0, 1500, 350, step=50,
                                           label="At a line break (ms)")
                    para_pause = gr.Slider(0, 3000, 600, step=100,
                                           label="At a blank line / paragraph (ms)",
                                           info="Short paragraphs and dialogue lines "
                                                "automatically get ~55% of this - they "
                                                "are line breaks, not scene changes")
                    lead_ms = gr.Slider(0, 2000, 350, step=50,
                                        label="Silence before the first word (ms)")
                    tail_ms = gr.Slider(0, 3000, 900, step=50,
                                        label="Silence after the last word (ms)",
                                        info="Chunks are trimmed so the gaps "
                                             "between them are exact, which left "
                                             "finished files ending 0-160ms after "
                                             "the final syllable - the file simply "
                                             "stopped. This is the room to breathe "
                                             "at the very edges.")
                    spara_pause = gr.Slider(
                        0, 2000, 320, step=20,
                        label="At a SHORT paragraph / dialogue line (ms)",
                        info="These scripts put every spoken line in its own "
                             "paragraph. At the full 800ms a rapid exchange read "
                             "as unrelated statements; 320ms reads as a breath.")
                    gr.Markdown(
                        "Punctuation is passed through untouched, so it still does "
                        "its work: `.` short pause · `,` tiny · `...` dramatic · "
                        "`—` beat · `!` energy · `?` rising tone."
                    )

                with gr.Accordion("Prosody chaining", open=False):
                    gr.Markdown(
                        "F5-TTS generates a **continuation** of whatever audio it "
                        "is conditioned on. Feeding it the same reference clip for "
                        "every chunk restarts the intonation 200 times a story - "
                        "which is what made narration sound like separate "
                        "recordings. Chaining conditions each chunk on the previous "
                        "one instead.\n\n"
                        "*Anchored*: your real clip is concatenated **before** the "
                        "previous chunk rather than replaced by it, so genuine "
                        "human audio stays in the window. Measured voice distance - "
                        "no chaining `0.0116`, pure chaining `0.0123`, "
                        "anchored **`0.0115`**."
                    )
                    chain = gr.Checkbox(True, label="Chain prosody across chunks")
                    chain_across_para = gr.Checkbox(
                        True, label="Keep chaining across paragraph breaks",
                        info="A narrator pauses at a paragraph; they do not reset "
                             "their voice. Re-anchoring here dropped chaining to "
                             "45-70% depending on how the author formatted "
                             "paragraphs, and tracked perceived quality exactly.")
                    chain_reanchor = gr.Slider(
                        1, 20, 8, step=1, label="Re-anchor every N chunks",
                        info="Return to the real reference clip this often. Lower "
                             "is safer for timbre, higher is smoother.")
                    drift_limit = gr.Slider(
                        0.0, 0.15, 0.045, step=0.005,
                        label="Re-anchor if voice drifts past",
                        info="0 disables. Unchained generation reaches 0.041 "
                             "naturally, so below that it fires constantly.")

                with gr.Accordion("Post-processing", open=False):
                    gr.Markdown(
                        "Vocos decodes accurate speech that is also dry and thin. "
                        "This is a gentle mastering chain: +2.5 dB low shelf at "
                        "200 Hz, a 2nd-order roll-off from 11 kHz, and 2:1 "
                        "compression. The unprocessed mix is **always** kept "
                        "alongside as `<name>_raw.wav`, so it is reversible and an "
                        "A/B never needs a re-render."
                    )
                    apply_warmth = gr.Checkbox(
                        True, label="Warmth, de-harsh and compression")

                with gr.Accordion("Reproducibility", open=False):
                    seed_num = gr.Number(
                        7, label="Seed", precision=0,
                        info="Fixed so a run repeats and a bad chunk can be "
                             "re-rendered deliberately. F5-TTS otherwise draws a "
                             "fresh random seed per chunk.")
                    lock_seed = gr.Checkbox(
                        True, label="Same seed for every chunk",
                        info="Off uses seed+i. The probes locked it and came out "
                             "consistently cleaner.")
                gr.Markdown(
                    "**Speed levers**: lower NFE, bigger chunks, and a *shorter* "
                    "reference clip (every chunk re-synthesises the reference, so a "
                    "6 s clip beats a 12 s one).\n\n"
                    f"Every chunk is saved to `{PARTS_DIR}` as it finishes, so a long "
                    "run can never lose all its work again."
                )

    with gr.Tab("Queue"):
        gr.Markdown(
            "### Overnight batch\n"
            "Paste a story in the **Generate** tab, give it a title, then "
            "**Add to queue** here. Repeat for as many stories as you like and "
            "press **Run entire queue** before bed.\n\n"
            "- Uses the checkpoint, reference clip and settings selected in the "
            "Generate tab.\n"
            "- A story that fails is moved to `queue/failed/` and the run "
            "**continues** — one bad script never costs you the night.\n"
            "- Finished sources move to `queue/done/`, so nothing is generated "
            "twice.\n"
            f"- Audio lands in `{OUT_DIR}` named after each title.\n\n"
            "For a truly unattended run, close the browser and use "
            "`run_queue.bat` instead — it needs no browser tab open."
        )
        with gr.Row():
            q_title = gr.Textbox(label="Title for the script currently in the "
                                       "Generate tab", placeholder="munjya")
            q_add = gr.Button("➕ Add to queue", variant="primary")
        q_table = gr.Dataframe(value=queue_table(), headers=["queued file", "size"],
                               datatype=["str", "str"], interactive=False,
                               label="Pending")
        with gr.Row():
            q_run = gr.Button("▶ Run entire queue", variant="primary")
            q_refresh = gr.Button("↻ Refresh")
            q_clear = gr.Button("🗑 Clear pending")
        q_log = gr.Textbox(label="Queue log", lines=16, interactive=False,
                           elem_id="qlog")

    with gr.Tab("Reference clips"):
        gr.Markdown(
            "Add your own reference clips. Keep one per mood — calm narration, "
            "tense, dialogue — and switch between them; this is your main control "
            "over delivery.\n\n"
            "**6–10 seconds** is the sweet spot. The transcript must be the exact "
            "words spoken."
        )
        with gr.Row():
            with gr.Column():
                up = gr.Audio(label="Upload or record a clip", type="filepath",
                              sources=["upload", "microphone"])
                ref_name = gr.Textbox(label="Save as (name)", placeholder="calm_narration")
                new_txt = gr.Textbox(label="Transcript of this clip", lines=4)
                with gr.Row():
                    auto_btn = gr.Button("Auto-transcribe (Whisper, slow)")
                    save_btn = gr.Button("Save reference", variant="primary")
                save_msg = gr.Textbox(label="", lines=2, interactive=False)
            with gr.Column():
                gr.Markdown(
                    "**Why the transcript matters:** the model aligns what it hears in "
                    "the clip with what it reads. A wrong word makes it mis-model your "
                    "voice. Auto-transcribe is a starting point — always correct it by "
                    "hand (Whisper misheard `वाजता` as `वास्ता` on your first clip)."
                )

    with gr.Tab("Pronunciation"):
        gr.Markdown(
            "The model reads **characters**, so the way to fix a mispronounced word "
            "is to respell it the way you want it said. Entries are whole-word "
            "replacements applied before synthesis.\n\n"
            "This is the practical handle for the Marathi च/ज problem — the dental "
            "*ts/dz* versus palatal *ch/j* distinction that Devanagari does not write. "
            "If a word comes out with the wrong one, try alternative spellings until it "
            "lands, then save it here so it is fixed everywhere.\n\n"
            "*Example:* `चमचा → चमचा` (replace with a spelling that produced the sound "
            "you wanted). Test with a one-line script — that is far faster than "
            "re-running a whole story."
        )
        auto_translit = gr.Checkbox(
            True,
            label="Auto-transliterate English words to Devanagari (no setup needed)",
            info="Uses the CMU pronouncing dictionary, so it follows how a word is "
                 "SAID, not how it is spelled: buyer → बायर. Your entries below "
                 "always override it.")
        use_dict = gr.Checkbox(True, label="Apply pronunciation dictionary (overrides)")
        dict_rows = gr.Dataframe(
            value=dict_to_rows(load_dict()) or [["", ""]],
            headers=["word as written", "replace with"],
            datatype=["str", "str"], row_count=(1, "dynamic"), column_count=(2, "fixed"),
            label="Replacements",
        )
        gr.Markdown(
            "### English words\n"
            "IndicF5 is trained on Indic script, so Latin words are out of "
            "distribution — they come out mangled or get skipped. Respell them in "
            "Devanagari (`buyer → बायर`) and they are read as sounds the model knows. "
            "**Scan** pulls every English word out of your script and pre-fills the "
            "common ones; fill the rest in yourself. Acronyms work best spaced out "
            "letter by letter: `IT → आय टी`."
        )
        with gr.Row():
            scan_btn = gr.Button("🔍 Scan script for English words")
            fill_btn = gr.Button("✨ Auto-fill blanks")
            dict_save = gr.Button("Save dictionary", variant="primary")
            dict_reload = gr.Button("Reload from disk")
        dict_msg = gr.Textbox(label="", lines=1, interactive=False)
        test_in = gr.Textbox(label="Preview text", lines=2,
                             placeholder="Paste a line to see the substitutions applied")
        test_out = gr.Textbox(label="After substitution", lines=2, interactive=False)
        test_btn = gr.Button("Preview")

    # wiring
    ref_wav.change(lambda p: ref_text_for(p), ref_wav, ref_txt)
    refresh.click(lambda: (gr.update(choices=list_ckpts()), gr.update(choices=list_refs())),
                  None, [ckpt, ref_wav])
    est_btn.click(estimate, [script, max_chars, nfe, ref_wav], est)
    mood.change(lambda m: gr.update(value=float(re.search(r"([\d.]+)", m).group(1)))
                if re.search(r"([\d.]+)", m or "") else gr.update(),
                mood, speed)
    go.click(generate,
             [script, ckpt, ref_wav, ref_txt, speed, nfe, pause_ms, max_chars,
              use_dict, dict_rows, auto_translit, device_choice,
              line_pause, para_pause, title, cfg, trim, drop_dir,
              expand_nums, one_pass, sent_pause, pace, fit_dur, per_sent,
              max_secs, lead_ms, tail_ms, flow_pause,
              spara_pause, chain, chain_across_para, chain_reanchor,
              drift_limit, apply_warmth, seed_num, lock_seed],
             [audio_out, status])

    q_add.click(add_to_queue, [q_title, script], [q_table, q_log])
    q_refresh.click(lambda: (queue_table(), "Refreshed."), None, [q_table, q_log])
    q_clear.click(clear_queue, None, [q_table, q_log])
    q_run.click(run_queue,
                [ckpt, ref_wav, ref_txt, speed, nfe, pause_ms, max_chars,
                 use_dict, dict_rows, auto_translit, device_choice,
                 line_pause, para_pause, cfg, trim, drop_dir,
                 expand_nums, one_pass, sent_pause, pace, fit_dur, per_sent,
              max_secs, lead_ms, tail_ms, flow_pause,
                 spara_pause, chain, chain_across_para, chain_reanchor,
                 drift_limit, apply_warmth, seed_num, lock_seed],
                [q_table, q_log])

    auto_btn.click(autotranscribe, up, new_txt)
    save_btn.click(save_reference, [up, ref_name, new_txt], [save_msg, ref_wav, ref_txt])

    scan_btn.click(scan_english, [script, dict_rows], [dict_rows, dict_msg])
    fill_btn.click(autofill_blanks, dict_rows, [dict_rows, dict_msg])
    dict_save.click(lambda rows: (save_dict(rows_to_dict(rows)), "Saved.")[1],
                    dict_rows, dict_msg)
    dict_reload.click(lambda: (dict_to_rows(load_dict()) or [["", ""]], "Reloaded."),
                      None, [dict_rows, dict_msg])
    test_btn.click(
        lambda t, rows, ud, at: prepare_text(t, rows_to_dict(rows), ud, at),
        [test_in, dict_rows, use_dict, auto_translit], test_out)

if __name__ == "__main__":
    print(f"device: {device_label()}")
    print(f"models: {MODELS_DIR}\nrefs  : {REF_DIR}\nout   : {OUT_DIR}")
    demo.queue().launch(
        theme=STUDIO_THEME, css=STUDIO_CSS,
        server_name="127.0.0.1", server_port=7860, inbrowser=True,
        # gradio refuses paths outside cwd/temp unless explicitly allowed
        allowed_paths=[str(OUT_DIR), str(REF_DIR), str(MODELS_DIR)],
    )
