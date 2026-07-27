# -*- coding: utf-8 -*-
"""
English -> Devanagari transliteration for Marathi TTS.

Pronunciation-based, not spelling-based: look the word up in the CMU
pronouncing dictionary (~126k words) and compose the phonemes into Devanagari
syllables. That is why 'buyer' becomes बायर and not बुयेर - English spelling
is irregular, its pronunciation is not.

Falls back to letter rules for words CMU has never seen (names, coinages).
"""
import re
from functools import lru_cache

try:
    import cmudict as _cmudict
    _CMU = _cmudict.dict()
except Exception:                       # library missing - rules only
    _CMU = {}

# ---------------------------------------------------------------- tables ---
CONS = {
    "P": "प", "B": "ब", "T": "ट", "D": "ड", "K": "क", "G": "ग",
    "CH": "च", "JH": "ज", "F": "फ", "V": "व्ह", "TH": "थ", "DH": "द",
    "S": "स", "Z": "झ", "SH": "श", "ZH": "झ", "HH": "ह",
    "M": "म", "N": "न", "NG": "ंग", "L": "ल", "R": "र", "W": "व", "Y": "य",
}

# vowel -> (independent form, matra to hang off a consonant)
VOW = {
    "AA": ("आ", "ा"), "AE": ("ॲ", "ॅ"), "AH": ("अ", ""), "AO": ("ऑ", "ॉ"),
    "AW": ("औ", "ौ"), "AY": ("आय", "ाय"), "EH": ("ए", "े"), "ER": ("अर", "र"),
    "EY": ("ए", "े"), "IH": ("इ", "ि"), "IY": ("ई", "ी"), "OW": ("ओ", "ो"),
    "OY": ("ऑय", "ॉय"), "UH": ("उ", "ु"), "UW": ("ऊ", "ू"),
}

LETTER = {
    "A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ", "G": "जी",
    "H": "एच", "I": "आय", "J": "जे", "K": "के", "L": "एल", "M": "एम", "N": "एन",
    "O": "ओ", "P": "पी", "Q": "क्यू", "R": "आर", "S": "एस", "T": "टी", "U": "यू",
    "V": "व्ही", "W": "डब्ल्यू", "X": "एक्स", "Y": "वाय", "Z": "झेड",
}

HALANT = "्"
_STRESS = re.compile(r"\d")


def _is_vowel(ph):
    return ph in VOW


def phonemes_to_devanagari(phones):
    """Compose ARPAbet phonemes into Devanagari, respecting the abugida rules."""
    ph = [_STRESS.sub("", p).upper() for p in phones]
    # A schwa after a consonant is simply that consonant's inherent vowel in
    # Devanagari, so it must NOT be dropped - only a schwa after a vowel is
    # spurious (सायंटिफिक, not सायअंटिफिक).
    cleaned = []
    for i, p in enumerate(ph):
        if p == "AH" and i > 0 and _is_vowel(ph[i - 1]):
            continue
        cleaned.append(p)
    ph = cleaned

    out, i = [], 0
    while i < len(ph):
        p = ph[i]
        if _is_vowel(p):
            # r-coloured vowel right after another vowel is just र (बायर)
            if p == "ER" and i > 0:
                out.append("र")
            else:
                out.append(VOW[p][0])                  # standalone vowel
            i += 1
            continue
        if p in CONS:
            nxt = ph[i + 1] if i + 1 < len(ph) else None
            # n/m before another consonant -> anusvara, the Marathi convention
            if p in ("N", "M") and nxt and nxt in CONS:
                out.append("ं")
                i += 1
                continue
            if nxt and _is_vowel(nxt):
                matra = VOW[nxt][1]
                out.append(CONS[p] + matra)
                i += 2
            else:
                # final consonant keeps its inherent vowel; mid-word gets halant
                out.append(CONS[p] + ("" if i == len(ph) - 1 else HALANT))
                i += 1
            continue
        i += 1
    return "".join(out)


# ------------------------------------------------------ spelling fallback --
_RULES = [
    ("tion", "शन"), ("sion", "शन"), ("ough", "औ"), ("ight", "ाइट"),
    ("ch", "च"), ("sh", "श"), ("th", "थ"), ("ph", "फ"), ("ck", "क"),
    ("oo", "ू"), ("ee", "ी"), ("ea", "ी"), ("ai", "ै"), ("ou", "ौ"),
    ("aw", "ॉ"), ("ay", "े"), ("ey", "े"), ("qu", "क्व"),
    ("a", "ॅ"), ("e", "े"), ("i", "ि"), ("o", "ो"), ("u", "ु"),
    ("b", "ब"), ("c", "क"), ("d", "ड"), ("f", "फ"), ("g", "ग"), ("h", "ह"),
    ("j", "ज"), ("k", "क"), ("l", "ल"), ("m", "म"), ("n", "न"), ("p", "प"),
    ("r", "र"), ("s", "स"), ("t", "ट"), ("v", "व्ह"), ("w", "व"),
    ("x", "क्स"), ("y", "य"), ("z", "झ"),
]


def _by_spelling(word):
    w, out = word.lower(), []
    while w:
        for src, dst in _RULES:
            if w.startswith(src):
                out.append(dst)
                w = w[len(src):]
                break
        else:
            w = w[1:]
    return "".join(out)


def _is_acronym(word):
    # ALL-CAPS short tokens are acronyms even when they collide with a real
    # word ('IT' is आय टी, not इट).
    return word.isupper() and 2 <= len(word) <= 5


@lru_cache(maxsize=4096)
def transliterate(word):
    """English word -> Devanagari. Returns '' for anything unusable."""
    w = (word or "").strip().strip(".,;:!?'\"-")
    if not w or not re.match(r"^[A-Za-z][A-Za-z'\-\.]*$", w):
        return ""
    if _is_acronym(w):                       # IT -> आय टी
        return " ".join(LETTER[c] for c in w if c in LETTER)
    entry = _CMU.get(w.lower())
    if entry:
        return phonemes_to_devanagari(entry[0])
    return _by_spelling(w)


if __name__ == "__main__":
    tests = ["buyer", "Buyer", "Scientific", "Rational", "Netflix", "camping",
             "software", "engineer", "highway", "torch", "hotel", "online",
             "mobile", "IT", "EMI", "OYO", "RTO", "WhatsApp", "Instagram",
             "flat", "room", "app", "caption", "trekking", "Google"]
    for t in tests:
        print(f"  {t:12} -> {transliterate(t)}")
