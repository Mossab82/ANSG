"""Parsing of night numbers written as digits or Arabic ordinal words.

Supports the feminine ordinal constructions used in night headings of
printed editions of *Alf Layla wa-Layla*, e.g.::

    الليلة الأولى                       -> 1
    الليلة الحادية عشرة                  -> 11
    الليلة الخامسة والعشرون              -> 25
    الليلة الحادية والعشرون بعد المائة    -> 121
    الليلة الثالثة بعد الثلاثمائة         -> 303
    الليلة الأولى بعد الألف              -> 1001
    الليلة ٢٥ / الليلة 25                -> 25

The parser is strict: any unrecognised word yields ``None`` rather than a
guess. Spelling variants (hamza/alef, ta marbuta/ha, ya/alef maqsura, مائة/مئة)
are folded before lookup.
"""

from __future__ import annotations

import re

_UNITS = {
    "اول": 1, "اولي": 1, "حاديه": 1, "حادي": 1, "واحد": 1, "واحده": 1,
    "ثانيه": 2, "ثاني": 2, "اثنان": 2, "اثنين": 2, "اثنتان": 2, "اثنتين": 2,
    "ثالثه": 3, "ثالث": 3,
    "رابعه": 4, "رابع": 4,
    "خامسه": 5, "خامس": 5,
    "سادسه": 6, "سادس": 6,
    "سابعه": 7, "سابع": 7,
    "ثامنه": 8, "ثامن": 8,
    "تاسعه": 9, "تاسع": 9,
    "عاشره": 10, "عاشر": 10,
}
_TEEN = {"عشره", "عشر"}
_TENS = {
    "عشرون": 20, "عشرين": 20, "ثلاثون": 30, "ثلاثين": 30, "اربعون": 40, "اربعين": 40,
    "خمسون": 50, "خمسين": 50, "ستون": 60, "ستين": 60, "سبعون": 70, "سبعين": 70,
    "ثمانون": 80, "ثمانين": 80, "تسعون": 90, "تسعين": 90,
}
_CARDINAL_STEM = {
    "ثلاث": 3, "اربع": 4, "خمس": 5, "ست": 6, "سبع": 7, "ثمان": 8, "ثماني": 8, "تسع": 9,
}
_HUNDRED = {"مايه", "ميه"}
_TWO_HUNDRED = {"مايتين", "مايتان", "ميتين", "ميتان"}
_THOUSAND = {"الف"}
_FILLER = {"بعد", "و"}

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_DIACRITICS = re.compile(r"[ً-ْٰـ]")


def _fold(word: str) -> str:
    w = _DIACRITICS.sub("", word)
    w = re.sub("[أإآٱ]", "ا", w)
    w = w.replace("ة", "ه").replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و")
    return w


def _strip_prefixes(w: str) -> str:
    if w.startswith("و") and len(w) > 2 and _strip_article(w[1:]) != w[1:]:
        w = w[1:]
    elif w.startswith("و") and _lookup_ok(w[1:]) and not _lookup_ok(w):
        w = w[1:]
    return _strip_article(w)


def _strip_article(w: str) -> str:
    return w[2:] if w.startswith("ال") and len(w) > 3 else w


def _lookup_ok(w: str) -> bool:
    w = _strip_article(w)
    return (
        w in _UNITS or w in _TEEN or w in _TENS or w in _CARDINAL_STEM
        or w in _HUNDRED or w in _TWO_HUNDRED or w in _THOUSAND or _compound_hundred(w) is not None
    )


def _compound_hundred(w: str) -> int | None:
    for stem, val in _CARDINAL_STEM.items():
        for h in _HUNDRED:
            if w == stem + h:
                return val * 100
    return None


def parse_arabic_number(text: str) -> int | None:
    """Parse a night number from digits or Arabic ordinal words; ``None`` if unsure."""
    t = text.strip().translate(_DIGITS)
    if re.fullmatch(r"\d+", t):
        return int(t)
    words = [_fold(w) for w in re.split(r"\s+", t) if w]
    if not words:
        return None
    total = 0
    unit = 0
    pending_stem: int | None = None
    for raw in words:
        if raw in _FILLER:
            continue
        w = _strip_prefixes(raw)
        if w in _FILLER:
            continue
        if pending_stem is not None:
            if w in _HUNDRED:
                total += pending_stem * 100
                pending_stem = None
                continue
            return None
        if w in _UNITS:
            if unit:
                return None
            unit = _UNITS[w]
        elif w in _TEEN:
            if not 1 <= unit <= 9:
                return None
            unit += 10
        elif w in _TENS:
            total += _TENS[w]
        elif w in _CARDINAL_STEM:
            pending_stem = _CARDINAL_STEM[w]
        elif w in _HUNDRED:
            total += 100
        elif w in _TWO_HUNDRED:
            total += 200
        elif w in _THOUSAND:
            total += 1000
        elif (h := _compound_hundred(w)) is not None:
            total += h
        else:
            return None
    if pending_stem is not None:
        return None
    value = total + unit
    return value if value > 0 else None
