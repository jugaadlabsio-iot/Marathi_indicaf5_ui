# -*- coding: utf-8 -*-
"""
Numbers -> Marathi words, for TTS.

The model reads characters. Digits are barely represented in the training data,
so "1994" or "३०४" come out wrong or get skipped. Spelling them out fixes it.

Rules chosen to match how a narrator actually reads:
  47            -> सत्तेचाळीस
  304           -> तीनशे चार
  2019          -> दोन हजार एकोणीस
  1994 (year)   -> एकोणीसशे चौऱ्याण्णव
  MH 09 BT 1994 -> एम एच शून्य नऊ बी टी एकोणीसशे चौऱ्याण्णव   (plate: digit-wise)
  ९:३०          -> नऊ तीस
"""
import re

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

ONES = [
    "शून्य", "एक", "दोन", "तीन", "चार", "पाच", "सहा", "सात", "आठ", "नऊ",
    "दहा", "अकरा", "बारा", "तेरा", "चौदा", "पंधरा", "सोळा", "सतरा", "अठरा",
    "एकोणीस", "वीस", "एकवीस", "बावीस", "तेवीस", "चोवीस", "पंचवीस", "सव्वीस",
    "सत्तावीस", "अठ्ठावीस", "एकोणतीस", "तीस", "एकतीस", "बत्तीस", "तेहतीस",
    "चौतीस", "पस्तीस", "छत्तीस", "सदतीस", "अडतीस", "एकोणचाळीस", "चाळीस",
    "एक्केचाळीस", "बेचाळीस", "त्रेचाळीस", "चव्वेचाळीस", "पंचेचाळीस",
    "सेहेचाळीस", "सत्तेचाळीस", "अठ्ठेचाळीस", "एकोणपन्नास", "पन्नास",
    "एक्कावन्न", "बावन्न", "त्रेपन्न", "चोपन्न", "पंचावन्न", "छप्पन्न",
    "सत्तावन्न", "अठ्ठावन्न", "एकोणसाठ", "साठ", "एकसष्ट", "बासष्ट", "त्रेसष्ट",
    "चौसष्ट", "पासष्ट", "सहासष्ट", "सदुसष्ट", "अडुसष्ट", "एकोणसत्तर", "सत्तर",
    "एक्काहत्तर", "बाहत्तर", "त्र्याहत्तर", "चौऱ्याहत्तर", "पंचाहत्तर",
    "शहात्तर", "सत्त्याहत्तर", "अठ्ठ्याहत्तर", "एकोणऐंशी", "ऐंशी",
    "एक्क्याऐंशी", "ब्याऐंशी", "त्र्याऐंशी", "चौऱ्याऐंशी", "पंच्याऐंशी",
    "शहाऐंशी", "सत्त्याऐंशी", "अठ्ठ्याऐंशी", "एकोणनव्वद", "नव्वद",
    "एक्याण्णव", "ब्याण्णव", "त्र्याण्णव", "चौऱ्याण्णव", "पंच्याण्णव",
    "शहाण्णव", "सत्त्याण्णव", "अठ्ठ्याण्णव", "नव्व्याण्णव",
]

# 100, 200 ... 900 - Marathi contracts these rather than saying "दोन शंभर"
HUNDREDS = ["", "शंभर", "दोनशे", "तीनशे", "चारशे", "पाचशे",
            "सहाशे", "सातशे", "आठशे", "नऊशे"]
# Years 1100-1999 are spoken as "<hundreds>शे <tens>". 2000+ are NOT -
# 2019 is दोन हजार एकोणीस, never वीसशे एकोणीस.
YEAR_HUNDREDS = {
    11: "अकराशे", 12: "बाराशे", 13: "तेराशे", 14: "चौदाशे", 15: "पंधराशे",
    16: "सोळाशे", 17: "सतराशे", 18: "अठराशे", 19: "एकोणीसशे",
}


def _under_100(n):
    return ONES[n]


def _under_1000(n):
    if n < 100:
        return _under_100(n)
    h, r = divmod(n, 100)
    out = HUNDREDS[h]
    return out if r == 0 else f"{out} {_under_100(r)}"


def number_to_words(n):
    """0 .. 99,99,999 in Marathi."""
    n = int(n)
    if n < 0:
        return "उणे " + number_to_words(-n)
    if n < 1000:
        return _under_1000(n)
    if n < 100000:
        th, r = divmod(n, 1000)
        out = f"{_under_1000(th)} हजार"
        return out if r == 0 else f"{out} {_under_1000(r)}"
    lakh, r = divmod(n, 100000)
    out = f"{_under_1000(lakh)} लाख"
    return out if r == 0 else f"{out} {number_to_words(r)}"


def year_to_words(n):
    """1994 -> एकोणीसशे चौऱ्याण्णव (how a year is actually spoken).
    2000+ falls through to दोन हजार ... which is how those are said."""
    n = int(n)
    head, tail = divmod(int(n), 100)
    if head in YEAR_HUNDREDS:
        if tail == 0:
            return YEAR_HUNDREDS[head]
        return f"{YEAR_HUNDREDS[head]} {_under_100(tail)}"
    return number_to_words(n)


def digits_to_words(s):
    """Digit by digit - for plates, phone numbers, room numbers."""
    return " ".join(ONES[int(c)] for c in s if c.isdigit())


_TOKEN = re.compile(r"[0-9०-९]+")
# a plate / code: letters immediately around digits (MH09, BT 1994, ३०४ A)
_PLATE_CTX = re.compile(
    r"(?:[A-Za-zऀ-ॿ]{1,3}\s*[-]?\s*)?[0-9०-९]{1,4}"
    r"(?:\s*[-]?\s*[A-Za-z]{1,3})?")


_TIME = re.compile(r"(?<=[0-9०-९])\s*[:：]\s*(?=[0-9०-९])")

# Symbols the model has no character for, so it simply skips them: "50%" came
# out as "पन्नास" with the percent silently dropped.
SYMBOLS = [
    ("%", " टक्के"), ("₹", "रुपये "), ("°", " अंश"),
    ("&", " आणि "), ("×", " गुणिले "), ("=", " बरोबर "), ("@", " ॲट "),
]
_RS = re.compile(r"\b(?:Rs\.?|INR)\s*", re.I)


def expand_symbols(text):
    """Say the symbols out loud.

    Runs AFTER the digits are spelled out, so by now "50%" is "पन्नास%" and
    only the sign is left to convert.
    """
    if not text:
        return text
    text = _RS.sub("रुपये ", text)
    for sym, word in SYMBOLS:
        text = text.replace(sym, word)
    return re.sub(r"[ \t]{2,}", " ", text)


def expand_numbers(text, plate_mode=False):
    """Replace every run of digits with Marathi words."""
    if not text:
        return text
    text = _TIME.sub(" ", text)          # 11:45 -> "11 45" so it reads as a time

    def repl(mo):
        raw = mo.group(0).translate(DEV_DIGITS)
        if not raw.isdigit():
            return mo.group(0)
        n = int(raw)
        # very long runs, or leading zeros -> digit by digit (phones, IDs, plates)
        if plate_mode or len(raw) > 6 or (len(raw) > 1 and raw[0] == "0"):
            return digits_to_words(raw)
        if len(raw) == 4 and 1100 <= n <= 1999:
            return year_to_words(n)      # 4-digit numbers in stories are usually years
        return number_to_words(n)

    return expand_symbols(_TOKEN.sub(repl, text))


if __name__ == "__main__":
    tests = ["47", "3", "10", "16", "28", "99", "100", "304", "1994", "2019",
             "2026", "1000", "12500", "150000", "09", "007"]
    for t in tests:
        print(f"  {t:8} -> {expand_numbers(t)}")
    print()
    for s in ["रूम नंबर ३०४ मध्ये.",
              "१६ ऑक्टोबर, २०१९ रोजी आग लागली.",
              "MH 09 BT 1994 ही गाडी होती.",
              "रात्रीचे ११:४५ वाजले होते.",
              "त्याचं वय अठ्ठावीस, पगार 45000 रुपये."]:
        print(f"  {s}\n   -> {expand_numbers(s)}\n")
