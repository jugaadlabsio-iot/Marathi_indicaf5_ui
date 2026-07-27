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
import re
import shutil
import time
import platform
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch

try:
    import translit                      # English -> Devanagari (CMU based)
except Exception:
    translit = None

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


def trim_silence(wav, sr, thresh_db=-40.0, keep_ms=40):
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


def prepare_text(text, d, use_dict, auto, drop_directions=True):
    """Your dictionary wins; auto-transliteration mops up whatever is left."""
    if drop_directions:
        text = strip_stage_directions(text)
    if use_dict:
        text = apply_dict(text, d)
    if auto:
        text = auto_translit_text(text)
    return text


# ------------------------------------------------------------- the model ---
_cache = {"key": None, "tts": None}


def get_tts(ckpt, vocab, device):
    key = (ckpt, vocab, device)
    if _cache["key"] == key:
        return _cache["tts"]
    from f5_tts.api import F5TTS
    tts = F5TTS(model="F5TTS_Base", ckpt_file=ckpt, vocab_file=vocab, device=device)
    # float16 makes this model emit NaN (flat DC = silence). Keep everything fp32.
    for name in ("ema_model", "model", "vocoder"):
        obj = getattr(tts, name, None)
        if obj is not None and hasattr(obj, "float"):
            setattr(tts, name, obj.float())
    _cache.update(key=key, tts=tts)
    return tts


# -------------------------------------------------------------- chunking ---
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


def split_blocks(text, max_chars=400):
    """Split while RESPECTING your line breaks.

    Collapsing all whitespace destroyed exactly the pauses you wrote into the
    script, which is why lines ran together. Each returned item is
    (chunk_text, pause_kind) where pause_kind says how much silence follows:
    'chunk' = mid-sentence-group, 'line' = you pressed Enter,
    'para'  = you left a blank line.
    """
    items = []
    for raw in (text or "").split("\n"):
        s = raw.strip()
        if not s:                                  # blank line -> longer pause
            if items:
                items[-1][1] = "para"
            continue
        parts = split_text(s, max_chars)
        for i, c in enumerate(parts):
            items.append([c, "chunk" if i < len(parts) - 1 else "line"])
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
def synthesize(script, ckpt, ref_wav, ref_txt, speed, nfe, max_chars,
               pauses, use_dict, dict_rows, auto_translit, device,
               out_name=None, on_progress=None, log=print,
               cfg=2.0, sway=-1.0, trim=True, drop_directions=True):
    """Core generation. Shared by the UI and the overnight queue runner."""
    vocab = str(MODELS_DIR / "vocab.txt")
    if not Path(vocab).exists():
        raise RuntimeError(f"vocab.txt missing from {MODELS_DIR}")

    text = prepare_text(script, rows_to_dict(dict_rows), use_dict, auto_translit,
                        drop_directions)
    items = split_blocks(text, int(max_chars))
    if not items:
        raise RuntimeError("Nothing to say.")

    tts = get_tts(ckpt, vocab, device)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = re.sub(r"[^A-Za-z0-9_\-]+", "_", out_name or "")[:40]
    run_dir = PARTS_DIR / (f"{tag}_{stamp}" if tag else stamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    pieces, sr, t0 = [], 24000, time.time()
    for i, (chunk, kind) in enumerate(items, 1):
        done = i - 1
        if on_progress:
            eta = ""
            if done:
                per = (time.time() - t0) / done
                eta = f" · ~{per*(len(items)-done)/60:.1f} min left"
            on_progress(done / len(items), f"Chunk {i}/{len(items)}{eta}")

        kw = dict(ref_file=ref_wav, ref_text=ref_txt.strip(),
                  speed=float(speed), nfe_step=int(nfe),
                  cfg_strength=float(cfg), sway_sampling_coef=float(sway),
                  remove_silence=False)   # we trim ourselves, see trim_silence
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
        if trim:
            wav = trim_silence(wav, sr)

        sf.write(run_dir / f"{i:04d}.wav", wav, sr)   # never lose a long run
        pieces.append(wav)
        if device == "cuda":
            torch.cuda.empty_cache()

        gap = pauses.get(kind, 0)
        if gap > 0 and i < len(items):
            pieces.append(np.zeros(int(sr * gap / 1000), dtype=np.float32))

    audio = np.concatenate(pieces)
    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = audio / peak * 0.95
    base = f"{tag}_{stamp}" if tag else f"story_{stamp}"
    out_path = OUT_DIR / f"{base}.wav"
    sf.write(out_path, audio, sr)
    return out_path, len(audio) / sr, time.time() - t0, len(items), run_dir


def generate(script, ckpt, ref_wav, ref_txt, speed, nfe, pause_ms, max_chars,
             use_dict, dict_rows, auto_translit, device_choice,
             line_pause, para_pause, title, cfg, trim, drop_dir,
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
    pauses = {"chunk": int(pause_ms), "line": int(line_pause),
              "para": int(para_pause), "end": 0}
    progress(0, desc=f"Loading model on {device}...")
    try:
        out_path, dur, took, n, run_dir = synthesize(
            script, ckpt, ref_wav, ref_txt, speed, nfe, max_chars, pauses,
            use_dict, dict_rows, auto_translit, device, out_name=title,
            on_progress=lambda f, d: progress(f, desc=d),
            cfg=cfg, trim=trim, drop_directions=drop_dir)
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
              progress=gr.Progress()):
    """Process every queued story. Designed to be left running overnight:
    one bad story never stops the rest, and finished work is never redone."""
    jobs = list_queue()
    if not jobs:
        return queue_table(), "Queue is empty."
    if not ckpt or not ref_wav or not (ref_txt or "").strip():
        return queue_table(), "Pick a checkpoint, a reference clip and its transcript first."

    device = DEVICE if device_choice == "auto" else device_choice
    pauses = {"chunk": int(pause_ms), "line": int(line_pause),
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
                cfg=cfg, trim=trim, drop_directions=drop_dir)
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
with gr.Blocks(title="Marathi Story Voice") as demo:
    gr.Markdown(f"## 🎙️ Marathi Story Voice\nRunning on **{device_label()}** · float32")

    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column(scale=3):
                script = gr.Textbox(label="Script (Marathi)", lines=18,
                                    placeholder="इथे तुमची भयकथा पेस्ट करा…")
                with gr.Row():
                    go = gr.Button("Generate audio", variant="primary", scale=2)
                    est_btn = gr.Button("Estimate time", scale=1)
                est = gr.Textbox(label="Estimate", lines=2, interactive=False)
                audio_out = gr.Audio(label="Result", type="filepath")
                status = gr.Textbox(label="Status", lines=4, interactive=False)

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
                                    label="NFE steps (16 draft · 32 production · 64 premium)")
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
                             "pause settings below are exact.")
                with gr.Accordion("Pauses", open=True):
                    gr.Markdown(
                        "Your **line breaks are respected**: press Enter for a "
                        "beat, leave a blank line for a longer one. Earlier "
                        "versions collapsed them, which is why lines ran together."
                    )
                    pause_ms = gr.Slider(0, 800, 150, step=25,
                                         label="Within a long line (ms)")
                    line_pause = gr.Slider(0, 1500, 350, step=50,
                                           label="At a line break (ms)")
                    para_pause = gr.Slider(0, 3000, 800, step=100,
                                           label="At a blank line / paragraph (ms)")
                    gr.Markdown(
                        "Punctuation is passed through untouched, so it still does "
                        "its work: `.` short pause · `,` tiny · `...` dramatic · "
                        "`—` beat · `!` energy · `?` rising tone."
                    )
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
        q_log = gr.Textbox(label="Queue log", lines=14, interactive=False)

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
              line_pause, para_pause, title, cfg, trim, drop_dir],
             [audio_out, status])

    q_add.click(add_to_queue, [q_title, script], [q_table, q_log])
    q_refresh.click(lambda: (queue_table(), "Refreshed."), None, [q_table, q_log])
    q_clear.click(clear_queue, None, [q_table, q_log])
    q_run.click(run_queue,
                [ckpt, ref_wav, ref_txt, speed, nfe, pause_ms, max_chars,
                 use_dict, dict_rows, auto_translit, device_choice,
                 line_pause, para_pause, cfg, trim, drop_dir],
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
        server_name="127.0.0.1", server_port=7860, inbrowser=True,
        # outputs live on D: - gradio refuses paths outside cwd/temp unless allowed
        allowed_paths=[str(OUT_DIR), str(REF_DIR), str(MODELS_DIR)],
    )
