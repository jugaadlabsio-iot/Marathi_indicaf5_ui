# -*- coding: utf-8 -*-
"""How long should this Devanagari text take to say?

WHY THIS EXISTS
---------------
F5-TTS decides, before it generates anything, how many mel frames the speech
gets. Left to itself it does this (f5_tts/infer/utils_infer.py):

    ref_text_len = len(ref_text.encode("utf-8"))
    gen_text_len = len(gen_text.encode("utf-8"))
    duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

It budgets time by **UTF-8 byte count**. For English that is roughly fine.
For Devanagari it is badly wrong, because every codepoint costs 3 bytes
whether or not it takes any time to say:

    कमल      3 codepoints,  9 bytes, 3 syllables ->  3.0 bytes per syllable
    शाळा     4 codepoints, 12 bytes, 2 syllables ->  6.0 bytes per syllable
    कोल्हापूर  9 codepoints, 27 bytes, 4 syllables ->  6.8 bytes per syllable

Matras, anusvara and the halant are separate codepoints that add 3 bytes each.
The halant actually makes a word *shorter* - it deletes the inherent vowel -
while still costing 3 bytes.

So a line of plain consonants gets barely half the time it needs, and a
matra-heavy line gets nearly double. That single fact explains the symptoms:

  under-allotted -> the model crams: rushed pace, fast start, sentences run
                    together with no room to breathe, final words clipped, and
                    in the worst case whole words simply dropped (शाळा)
  over-allotted  -> the model drawls, smears conjuncts, or fills the extra
                    frames with noise

The fix is to stop guessing from bytes. Count actual syllables, calibrate
against the reference clip's own measured speaking rate, add real time for
punctuation, and hand F5-TTS an explicit `fix_duration`.
"""
import re
import unicodedata

# --- Devanagari codepoint classes -------------------------------------------
_CONS = set(range(0x0915, 0x093A)) | set(range(0x0958, 0x0960)) | set(range(0x0979, 0x0980))
_VOW_IND = set(range(0x0904, 0x0915)) | {0x0960, 0x0961}
_MATRA = set(range(0x093E, 0x094D)) | {0x093A, 0x093B, 0x094E, 0x094F, 0x0962, 0x0963}
_VIRAMA = 0x094D
_NUKTA = 0x093C
_ANUSVARA = {0x0900, 0x0901, 0x0902, 0x0903}

# matras and independent vowels that are held longer than the inherent 'a'
_LONG = {
    0x093E, 0x0940, 0x0942, 0x0944, 0x0947, 0x0948, 0x094B, 0x094C,   # ा ी ू ॄ े ै ो ौ
    0x0906, 0x0908, 0x090A, 0x090F, 0x0910, 0x0913, 0x0914,           # आ ई ऊ ए ऐ ओ औ
}

# Cost in "moras" - relative time units, scaled so an ordinary short syllable
# is 1.0. These are not physics, they are a fit to how Marathi narration times
# out; what matters is that they are *proportional* to duration, unlike bytes.
_M_SHORT = 1.00
_M_LONG = 1.42
_M_NASAL = 0.22          # anusvara / candrabindu / visarga
_M_CLUSTER = 0.30        # each extra consonant welded into a conjunct

# Silence that punctuation is worth, in seconds. Budgeted explicitly so the
# model has somewhere to put the pause instead of steamrolling through it.
PUNCT_PAUSE = {
    "।": 0.34, ".": 0.34, "?": 0.36, "!": 0.36,
    ",": 0.16, ";": 0.22, ":": 0.22,
    "—": 0.30, "–": 0.24, "…": 0.42,
}
_ELLIPSIS = re.compile(r"\.\.\.|…")


def moras(text):
    """Time-weighted syllable count for a Devanagari string.

    An akshara is a consonant cluster plus its vowel. A consonant starts a new
    syllable unless a virama has just deleted its vowel, in which case it is
    welded onto the one being built.
    """
    total = 0.0
    pending_virama = False
    cluster_extra = 0
    open_syl = False          # a syllable is being built and not yet counted

    def close():
        nonlocal total, cluster_extra, open_syl
        if open_syl:
            total += cluster_extra * _M_CLUSTER
        cluster_extra = 0
        open_syl = False

    for ch in unicodedata.normalize("NFC", text or ""):
        cp = ord(ch)
        if cp in _CONS:
            if pending_virama:
                cluster_extra += 1          # part of the same conjunct
            else:
                close()
                total += _M_SHORT           # assume inherent 'a' until a matra says otherwise
                open_syl = True
            pending_virama = False
        elif cp == _VIRAMA:
            pending_virama = True
        elif cp in _MATRA:
            if open_syl:
                total += (_M_LONG - _M_SHORT) if cp in _LONG else 0.0
            pending_virama = False
        elif cp in _VOW_IND:
            close()
            total += _M_LONG if cp in _LONG else _M_SHORT
            open_syl = True
            pending_virama = False
        elif cp in _ANUSVARA:
            total += _M_NASAL
            pending_virama = False
        elif cp == _NUKTA:
            pass                            # spelling only, no time
        elif ch.isalpha():
            # stray Latin that escaped transliteration - approximate
            total += 0.5
            pending_virama = False
        else:
            close()
            pending_virama = False
    close()
    return total


def punctuation_time(text):
    """Seconds of silence the punctuation in this text is asking for."""
    t = 0.0
    body = _ELLIPSIS.sub("…", text or "")
    for ch in body:
        t += PUNCT_PAUSE.get(ch, 0.0)
    return t


def speech_rate(ref_text, ref_seconds):
    """Moras per second, measured from the reference clip.

    This is what makes the estimate self-calibrating: it is the user's own
    narration speed, not a constant borrowed from some other language.
    """
    m = moras(ref_text)
    if m <= 0 or ref_seconds <= 0:
        return 6.0                       # sane Marathi narration fallback
    p = punctuation_time(ref_text)
    speaking = max(0.4 * ref_seconds, ref_seconds - p)
    return m / speaking


def estimate_seconds(text, rate, roominess=1.08, tail=0.35):
    """How long this chunk should be allowed to take.

    `roominess` is deliberate slack. Too little and the model crams or drops
    words; too much and it drawls or fills the gap with noise. 1.08 is a small
    cushion. `tail` keeps the final consonant from being cut off.
    """
    m = moras(text)
    if m <= 0:
        return max(0.35, tail)
    return m / max(rate, 0.5) * roominess + punctuation_time(text) + tail


def rate_report(ref_text, ref_seconds):
    """One line for the log, so a bad reference transcript is visible."""
    r = speech_rate(ref_text, ref_seconds)
    return (f"reference: {moras(ref_text):.1f} moras in {ref_seconds:.2f}s "
            f"-> {r:.2f} moras/sec")


if __name__ == "__main__":
    for w in ["कमल", "शाळा", "कोल्हापूर", "थांबले", "थांबला", "ळ", "ल",
              "मुंबई", "विद्यार्थी", "स्वप्न"]:
        b = len(w.encode("utf-8"))
        print(f"  {w:12} {b:3d} bytes  {moras(w):5.2f} moras  "
              f"{b/max(moras(w),.01):5.2f} bytes/mora")
    print()
    ref = "रात्री बारा वाजता पाटलांच्या पडवीत ते लॅपटॉपवर अहवाल लिहीत बसले होते."
    r = speech_rate(ref, 5.6)
    print(" ", rate_report(ref, 5.6))
    for s in ["शाळा बंद होती.",
              "कोल्हापूरच्या त्या जुन्या वाड्यात कोणीच राहत नव्हतं.",
              "तो थांबले होते. मग तो पुढे गेला."]:
        print(f"  {estimate_seconds(s, r):5.2f}s   {s}")
