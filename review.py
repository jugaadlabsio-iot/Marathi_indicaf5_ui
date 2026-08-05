# -*- coding: utf-8 -*-
"""Listen to a finished story, flag what is wrong, repair it.

    python review.py     ->  http://127.0.0.1:7861

You listen to the stitched file the way you normally would. When something
sounds wrong you press Flag; the page reads the player's current position,
works out which chunk is playing, and adds it to a list with its text. When
you are done, Repair re-renders exactly those chunks and rebuilds the story.

The timestamp -> chunk mapping is exact, not estimated: every chunk wav is on
disk, the pauses between them are known, so the timeline reconstructs to the
sample. Verified against three finished stories at 0.000s drift.

Runs on port 7861 so it can sit alongside the main app on 7860. The model is
only loaded when you actually repair something - browsing and listening do
not touch the GPU, so this is safe to use while a queue run is going.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import importlib.util
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mtts_app", HERE / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

PARTS = app.PARTS_DIR
OUT = app.OUT_DIR
# Must match run_queue.py's defaults or the reconstructed timeline drifts
# and every flagged position lands on the wrong chunk.
DEFAULT_PAUSES = {"chunk": 250, "flow": 60, "sent": 300, "line": 350,
                  "spara": 320, "para": 600, "end": 0}


def list_runs():
    """Runs that can be reviewed: indexed, and with a stitched wav to play."""
    out = []
    if PARTS.exists():
        for d in sorted(PARTS.iterdir(), reverse=True):
            if d.is_dir() and (d / "chunks.tsv").exists() and (OUT / f"{d.name}.wav").exists():
                out.append(d.name)
    return out


def read_index(run):
    rows = []
    f = PARTS / run / "chunks.tsv"
    for line in f.read_text(encoding="utf-8").splitlines()[1:]:
        parts = (line.split("\t") + ["", "", "", ""])[:4]
        rows.append(dict(file=parts[0], kind=parts[1],
                         planned=float(parts[2] or 0), text=parts[3]))
    return rows


def build_timeline(run, pauses=None, lead_ms=350):
    """(start, end, row) for every chunk, in the stitched file's own clock."""
    pauses = pauses or DEFAULT_PAUSES
    rows = read_index(run)
    d = PARTS / run
    t = lead_ms / 1000.0
    tl = []
    for i, r in enumerate(rows):
        p = d / r["file"]
        if not p.exists():
            tl.append((t, t, r))
            continue
        dur = sf.info(str(p)).duration
        tl.append((t, t + dur, r))
        t += dur
        if i < len(rows) - 1:
            t += pauses.get(r["kind"], 0) / 1000.0
    return tl


def chunk_at(tl, seconds):
    """Which chunk is playing at this moment? Falls to the nearest if the
    moment lands inside a pause."""
    for i, (a, b, r) in enumerate(tl, 1):
        if a <= seconds <= b:
            return i, r
    best, bi = None, 1
    for i, (a, b, r) in enumerate(tl, 1):
        d = min(abs(seconds - a), abs(seconds - b))
        if best is None or d < best:
            best, bi, br = d, i, r
    return (bi, tl[bi - 1][2]) if tl else (0, None)


# --- flags live on disk, not in a gradio component ---------------------------
# The first version kept them in a Dataframe and passed it in and out of every
# handler. Between the javascript bridge and the dataframe round trip, a click
# could fail with nothing in the log. A json file next to the chunks is boring,
# survives a page refresh, and can be read by the repair tool directly.
import json
import sys
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def flags_path(run):
    return PARTS / run / "flags.json"


def read_flags(run):
    p = flags_path(run)
    if not run or not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def write_flags(run, flags):
    flags = sorted({f["chunk"]: f for f in flags}.values(), key=lambda f: f["chunk"])
    flags_path(run).write_text(json.dumps(flags, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    return flags


def render_flags(run):
    flags = read_flags(run)
    if not flags:
        return "(nothing flagged yet)"
    out = [f"{len(flags)} flagged chunk(s)", ""]
    for f in flags:
        note = f"  [{f['note']}]" if f.get("note") else ""
        out.append(f"  chunk {f['chunk']:<4} at {f['at']}{note}")
        out.append(f"      {f.get('text','')}")
    out.append("")
    out.append("chunk numbers: " + ",".join(str(f["chunk"]) for f in flags))
    return "\n".join(out)


def load_run(run):
    if not run:
        return None, "(nothing flagged yet)", "Pick a story."
    wav = OUT / f"{run}.wav"
    tl = build_timeline(run)
    return (str(wav), render_flags(run),
            f"{run}\n{len(tl)} chunks · {sf.info(str(wav)).duration/60:.1f} min. "
            f"Play, and press Flag when something sounds wrong.")


def mm_ss(t):
    return f"{int(t)//60:d}:{int(t)%60:02d}"


def _nothing(run, msg):
    """Error return, shaped like the success one so gradio stays happy."""
    return render_flags(run) if run else "(nothing flagged yet)", msg, "", None, ""


def add_flag(run, seconds, note):
    """Flag a position, and load that chunk into the preview.

    Loading it straight away is the point: you hear the chunk on its own
    immediately after flagging it, so a mis-hit is obvious before you have
    moved on and forgotten what you heard.
    """
    if not run:
        return _nothing(run, "Pick a story first.")
    try:
        seconds = float(seconds or 0)
    except Exception:
        seconds = 0.0
    tl = build_timeline(run)
    if not tl:
        return _nothing(run, "That run has no chunk index.")
    idx, row = chunk_at(tl, seconds)
    if row is None:
        return _nothing(run, "Could not place that position.")

    flags = read_flags(run)
    if not any(f["chunk"] == idx for f in flags):
        flags.append(dict(chunk=idx, at=mm_ss(seconds), note=(note or "").strip(),
                          text=row["text"]))
        write_flags(run, flags)
        msg = f"Flagged chunk {idx} at {mm_ss(seconds)} — {row['text'][:60]}"
    else:
        msg = f"Chunk {idx} was already flagged — loaded it below."

    wav = PARTS / run / f"{idx:04d}.wav"
    return (render_flags(run), msg, str(idx),
            str(wav) if wav.exists() else None,
            show_around(run, seconds))          # numeric, not a string


def clear_flags(run):
    if run:
        write_flags(run, [])
    return "(nothing flagged yet)", "Cleared."


def undo_flag(run):
    flags = read_flags(run)
    if not flags:
        return render_flags(run), "Nothing to undo."
    gone = flags.pop()
    write_flags(run, flags)
    return render_flags(run), f"Removed chunk {gone['chunk']}."


def preview_chunk(run, which):
    """Play one chunk on its own, to confirm before repairing."""
    if not run:
        return None, "Pick a story first."
    try:
        idx = int(str(which).strip())
    except Exception:
        return None, "Give a chunk number."
    p = PARTS / run / f"{idx:04d}.wav"
    if not p.exists():
        return None, f"No chunk {idx} in this run."
    rows = read_index(run)
    text = rows[idx - 1]["text"] if 1 <= idx <= len(rows) else ""
    return str(p), f"Chunk {idx}: {text}"


def show_around(run, typed):
    """What is playing near this position? Answers 'which chunk was that?'.

    Accepts a number as well as a typed string. It has to: parse_at treats a
    dot as a minute separator, so handing it str(19.0) would read "19.0" as
    nineteen minutes.
    """
    t = typed if isinstance(typed, (int, float)) else parse_at(typed)
    if not run or t is None:
        return "Pick a story and type a position like 2:07."
    tl = build_timeline(run)
    out = []
    for a, b, r in tl:
        if b < t - 6 or a > t + 6:
            continue
        i = tl.index((a, b, r)) + 1
        here = "  <== here" if a <= t <= b else ""
        out.append(f"  {mm_ss(a)}-{mm_ss(b)}  chunk {i:<4} {r['text'][:52]}{here}")
    return "\n".join(out) or "Nothing near that position."


def repair(run, tries, pace, progress=gr.Progress()):
    idxs = [f["chunk"] for f in read_flags(run)]
    if not run or not idxs:
        return None, "Flag something first."

    import pacing
    index = read_index(run)
    ckpt = next((c for c in app.list_ckpts() if "slim" in c.lower()), None)
    if not ckpt:
        return None, "No checkpoint found."
    ref = min(app.list_refs(), key=lambda r: sf.info(r).duration)
    ref_txt = app.ref_text_for(ref)
    _, ref_norm, ref_sec = app.ref_profile(ref, ref_txt.strip())
    rate = pacing.speech_rate(ref_norm, ref_sec)
    room = min(app.MAX_ROOMINESS, 1.08 * float(pace))
    tts = app.get_tts(ckpt, str(app.MODELS_DIR / "vocab.txt"), app.DEVICE)

    log = []
    import shutil
    for n, idx in enumerate(idxs, 1):
        progress(n / (len(idxs) + 1), desc=f"Chunk {idx} ({n}/{len(idxs)})")
        if not (1 <= idx <= len(index)):
            log.append(f"chunk {idx}: out of range")
            continue
        r = index[idx - 1]
        w = PARTS / run / r["file"]
        orig = w.with_suffix(w.suffix + ".orig")
        if w.exists() and not orig.exists():
            shutil.copy(w, orig)               # never lose the original
        want = pacing.estimate_seconds(r["text"], rate, roominess=room)
        best, best_score, sr = None, -1.0, 24000
        for t in range(int(tries)):
            seed = 1097657232 + idx + t * 7919
            wav, sr, _ = tts.infer(
                gen_text=r["text"], ref_file=ref, ref_text=ref_txt.strip(),
                speed=1.0, nfe_step=32, cfg_strength=2.0,
                sway_sampling_coef=-1.0, remove_silence=False,
                seed=seed, fix_duration=ref_sec + want)
            wav = np.asarray(wav, dtype=np.float32)
            if np.isnan(wav).any():
                continue
            wav = app.trim_silence(wav, sr)
            bad = app.ends_mid_word(wav, sr) or app.starts_mid_word(wav, sr)
            score = -1.0 if bad else len(wav) / sr
            if score > best_score:
                best, best_score = wav, score
        if best is None:
            log.append(f"chunk {idx}: every take was clipped, kept the original")
            continue
        sf.write(w, app.fade_edges(best, sr), sr)
        log.append(f"chunk {idx}: replaced, {best_score:.2f}s — {r['text'][:44]}")

    # rebuild the story from whatever is on disk now
    progress(0.98, desc="Rebuilding the story")
    pieces, sr = [], 24000
    for i, r in enumerate(index, 1):
        p = PARTS / run / r["file"]
        if not p.exists():
            continue
        x, sr = sf.read(str(p))
        pieces.append(np.asarray(x, dtype=np.float32))
        gap = DEFAULT_PAUSES.get(r["kind"], 0)
        if gap and i < len(index):
            pieces.append(np.zeros(int(sr * gap / 1000), dtype=np.float32))
    audio = np.concatenate([np.zeros(int(sr * 0.35), np.float32),
                            np.concatenate(pieces),
                            np.zeros(int(sr * 0.9), np.float32)])
    audio = audio / (np.abs(audio).max() or 1.0) * 0.95
    out = OUT / f"{run}_repaired.wav"
    sf.write(str(out), audio, sr)
    log.append(f"\nrebuilt: {out.name}  ({len(audio)/sr/60:.1f} min)")
    return str(out), "\n".join(log)


def restore(run):
    import shutil
    n = 0
    index = read_index(run)
    for f in read_flags(run):
        idx = f["chunk"]
        if not (1 <= idx <= len(index)):
            continue
        w = PARTS / run / index[idx - 1]["file"]
        o = w.with_suffix(w.suffix + ".orig")
        if o.exists():
            shutil.copy(o, w)
            n += 1
    return f"Restored {n} chunk(s) to the original take."


# Read the player's position in the browser and put it in the `seconds` box.
#
# This deliberately returns ONE number and writes it to ONE output. The first
# version passed every input through the js function, including the flagged
# table; a dataframe does not survive that round trip, the call failed, and
# `seconds` silently fell back to 0 - so every flag landed on chunk 1.
#
# Takes the largest currentTime across all media elements: the page holds
# several players and the empty ones sit at 0.
# Where the playback position actually lives.
#
# Gradio's audio component does put an <audio> tag in the DOM, but it is a
# decoy: with a story loaded and playing it reports src "", readyState 0,
# duration null and currentTime 0. Playback runs through a WaveSurfer instance
# that is not reachable from the page's global scope. Reading currentTime -
# which is what the first three attempts at this did - therefore always
# returned 0, and every automatic flag landed on chunk 1.
#
# The player does render its position as text:
#     <time>0:19</time>  <time>10:47</time>     (current, duration)
#
# Several players share the page, so the pairs are grouped and the one with
# the longest duration is taken - that is the story, not a single chunk.
GRAB_TIME = """
(x) => {
  const toS = (s) => {
    const m = String(s || '').trim().match(/^(?:(\\d+):)?(\\d+):(\\d{2})/);
    if (!m) return null;
    return (parseInt(m[1] || 0) * 3600) + (parseInt(m[2]) * 60) + parseInt(m[3]);
  };
  const times = Array.from(document.querySelectorAll('time'))
                     .map(e => toS(e.textContent)).filter(v => v !== null);
  let cur = 0, best = -1;
  for (let i = 0; i + 1 < times.length; i += 2) {
    if (times[i + 1] > best) { best = times[i + 1]; cur = times[i]; }
  }
  if (best >= 0) return cur;
  const els = Array.from(document.querySelectorAll('audio, video'));
  let f = 0;
  for (const a of els) { const t = Number(a.currentTime); if (isFinite(t) && t > f) f = t; }
  return f;
}
"""


def parse_at(s):
    """A typed position -> seconds.

        0:19  0.19  ->  19 s      (a dot is a minute separator, not a decimal
        2:07  2.07  ->  127 s      point - people write timestamps both ways
        127         ->  127 s      and 0.19 always means nineteen seconds here)
    """
    s = str(s or "").strip().replace(",", ".")
    if not s:
        return None
    try:
        if ":" in s or "." in s:
            mins, _, secs = s.replace(".", ":").partition(":")
            return int(mins or 0) * 60 + float(secs or 0)
        return float(s)                       # a bare number is seconds
    except Exception:
        return None


def add_flag_at(run, typed, note):
    """Flag a position typed by hand, when the player's clock is not wanted."""
    t = parse_at(typed)
    if t is None:
        return _nothing(run, "Type a position like 2:07 or 2.07 (or seconds).")
    return add_flag(run, t, note)

with gr.Blocks(title="Story review") as demo:
    gr.Markdown(
        "# Story review\n"
        "Play the story. When something sounds wrong press **Flag** — the "
        "chunk that was playing is identified from the player's position and "
        "added below. **Repair** re-renders only those chunks and rebuilds "
        "the file.\n\n"
        "*The model is loaded only when you repair, so listening is safe "
        "while a queue run is going.*")

    with gr.Row():
        run = gr.Dropdown(list_runs(), label="Story", scale=4)
        refresh = gr.Button("Refresh list", scale=1)

    player = gr.Audio(label="Story", type="filepath", interactive=False)
    status = gr.Textbox(label="", lines=2, interactive=False)

    with gr.Row():
        note = gr.Textbox(label="What is wrong? (optional)",
                          placeholder="wrong word · clipped · rushed", scale=3)
        flag_btn = gr.Button("Flag this moment", variant="primary", scale=1)
    with gr.Row():
        seconds = gr.Number(value=0, label="Player position (s)", scale=1,
                            info="Filled in when you press Flag. If it stays "
                                 "at 0 the browser is not exposing the "
                                 "player's clock - use the box on the right.")
        typed_at = gr.Textbox(label="…or type the position", placeholder="0:21",
                              scale=1)
        flag_at_btn = gr.Button("Flag that position", scale=1)

    # The flagged chunk is loaded here the moment you flag it, so you can
    # confirm the hit while you still remember what you heard.
    with gr.Row():
        which = gr.Textbox(label="Chunk", scale=1)
        prev_btn = gr.Button("Play this chunk alone", scale=2)
    chunk_player = gr.Audio(label="The flagged chunk, on its own",
                            type="filepath", interactive=False)
    nearby = gr.Textbox(label="What is around it", lines=6, interactive=False)

    table = gr.Textbox(label="Flagged chunks (saved to flags.json in the run "
                             "folder, so a refresh does not lose them)",
                       lines=10, interactive=False, value="(nothing flagged yet)")

    with gr.Row():
        undo_btn = gr.Button("Remove last flag", scale=1)
        clear_btn = gr.Button("Clear all flags", scale=1)

    with gr.Accordion("Look up a position without flagging it", open=False):
        with gr.Row():
            look_at = gr.Textbox(label="What is playing at…", placeholder="2:07",
                                 scale=2)
            look_btn = gr.Button("Show chunks near that time", scale=1)

    gr.Markdown("---")
    with gr.Row():
        tries = gr.Slider(1, 8, 5, step=1, label="Takes per chunk",
                          info="Each take uses a different seed. The best "
                               "clean one is kept; a mispronounced word is "
                               "usually just an unlucky seed.")
        pace = gr.Slider(1.0, 1.35, 1.35, step=0.01, label="Roominess")
    with gr.Row():
        repair_btn = gr.Button("Repair flagged chunks and rebuild",
                               variant="primary", scale=3)
        restore_btn = gr.Button("Undo repairs on these chunks", scale=1)
    repaired = gr.Audio(label="Repaired story", type="filepath", interactive=False)
    rlog = gr.Textbox(label="Repair log", lines=8, interactive=False)

    refresh.click(lambda: gr.update(choices=list_runs()), None, run)
    run.change(load_run, run, [player, table, status])
    # js fills `seconds` first, then python reads it back - nothing but plain
    # strings and numbers ever crosses the javascript boundary
    # js receives the current value and returns the player's position, which
    # the identity fn then writes into `seconds`; fn=None with js= did not
    # reliably wire up here
    flag_out = [table, status, which, chunk_player, nearby]
    flag_btn.click(lambda t: t, seconds, seconds, js=GRAB_TIME).then(
        add_flag, [run, seconds, note], flag_out)
    flag_at_btn.click(add_flag_at, [run, typed_at, note], flag_out)
    typed_at.submit(add_flag_at, [run, typed_at, note], flag_out)
    undo_btn.click(undo_flag, run, [table, status])
    clear_btn.click(clear_flags, run, [table, status])
    look_btn.click(show_around, [run, look_at], nearby)
    look_at.submit(show_around, [run, look_at], nearby)
    prev_btn.click(preview_chunk, [run, which], [chunk_player, status])
    repair_btn.click(repair, [run, tries, pace], [repaired, rlog])
    restore_btn.click(restore, run, rlog)

if __name__ == "__main__":
    print("Review UI -> http://127.0.0.1:7861")
    demo.queue().launch(server_name="127.0.0.1", server_port=7861,
                        inbrowser=True,
                        allowed_paths=[str(OUT), str(PARTS), str(app.REF_DIR)])
