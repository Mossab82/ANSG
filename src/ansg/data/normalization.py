"""Configurable, offset-preserving Arabic text normalisation.

Design principles
-----------------
* **Lossless by default.** Classical narrative prose (e.g. rhymed *saj'*,
  vocalised verse, honorific ligatures) carries rhetorical and cultural
  information in diacritics, hamza spelling and orthographic detail. Every
  lossy operation (diacritic stripping, alef/ya/ta-marbuta folding) is
  therefore *opt-in*. The defaults only repair encoding artefacts
  (presentation forms, invisible bidi controls, whitespace runs).
* **Traceable.** :func:`normalize` returns, for every output character, the
  index of the raw character it came from, so annotations created on either
  representation can be projected onto the other.
* **Recorded.** The exact configuration is serialisable and hashed into run
  manifests.

The legacy ``Normalize_Text`` in ``Main.py``/``preprocess.py`` called
``pyarabic.araby.strip_harakat`` followed by ``strip_tashkeel`` unconditionally.
That behaviour is reproducible here with ``strip_diacritics=True`` but is no
longer the default (see ``docs/IMPLEMENTATION_GAPS.md``).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

TATWEEL = "ـ"
# Harakat, tanwin, shadda, sukun, maddah/hamza above/below combining marks, etc.
_HARAKAT = {chr(c) for c in range(0x064B, 0x0660)}
DAGGER_ALEF = "ٰ"  # superscript alef: orthographically meaningful, separate switch
_QURANIC_MARKS = {chr(c) for c in range(0x06D6, 0x06EE)} | {chr(c) for c in range(0x0610, 0x061B)}
# Invisible formatting/bidi controls frequently introduced by PDF/HTML extraction.
_INVISIBLE = {
    "​", "‌", "‍", "‎", "‏",
    "‪", "‫", "‬", "‭", "‮",
    "⁦", "⁧", "⁨", "⁩", "﻿", "­",
}
_ALEF_VARIANTS = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"}
_ARABIC_INDIC = {chr(0x0660 + i): str(i) for i in range(10)}
_EXT_ARABIC_INDIC = {chr(0x06F0 + i): str(i) for i in range(10)}
_WESTERN_TO_ARABIC_INDIC = {str(i): chr(0x0660 + i) for i in range(10)}


def _is_presentation_form(ch: str) -> bool:
    cp = ord(ch)
    return 0xFB50 <= cp <= 0xFDFF or 0xFE70 <= cp <= 0xFEFF


@dataclass(frozen=True)
class NormalizationConfig:
    """Normalisation switches. Defaults are non-destructive (see module doc)."""

    unicode_form: Literal["NFC", "NFKC", "none"] = "NFC"
    fold_presentation_forms: bool = True  # NFKC applied only to U+FB50–FDFF, U+FE70–FEFF
    remove_invisible_controls: bool = True
    collapse_whitespace: bool = True  # runs of spaces/tabs -> one space; keeps newlines
    max_consecutive_newlines: int | None = 2
    remove_tatweel: bool = True
    strip_diacritics: bool = False
    strip_dagger_alef: bool = False
    strip_quranic_marks: bool = False
    normalize_alef: bool = False
    normalize_alef_maqsura: bool = False  # ى -> ي
    normalize_teh_marbuta: bool = False  # ة -> ه
    digits: Literal["keep", "western", "arabic_indic"] = "keep"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "NormalizationConfig":
        data = dict(data or {})
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown normalization options: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def legacy(cls) -> "NormalizationConfig":
        """Approximation of the legacy ``Normalize_Text`` (diacritics removed)."""
        return cls(strip_diacritics=True, strip_dagger_alef=False, remove_tatweel=False,
                   fold_presentation_forms=False, remove_invisible_controls=False,
                   collapse_whitespace=False, max_consecutive_newlines=None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedText:
    """Normalised text plus a map from output indices to raw indices.

    ``offset_map`` has ``len(text) + 1`` entries; ``offset_map[i]`` is the raw
    index of the character that produced ``text[i]`` and the final entry is
    ``len(raw)``.
    """

    text: str
    raw: str
    offset_map: list[int] = field(repr=False)
    operations: dict[str, int] = field(default_factory=dict)

    def to_raw_span(self, start: int, end: int) -> tuple[int, int]:
        """Project a ``[start, end)`` span of normalised text onto the raw text."""
        if not 0 <= start <= end <= len(self.text):
            raise IndexError(f"span ({start}, {end}) outside normalised text")
        raw_start = self.offset_map[start]
        if end == start:
            return raw_start, raw_start
        # end is exclusive: take the raw index after the last contributing char
        last = self.offset_map[end - 1]
        nxt = self.offset_map[end]
        raw_end = nxt if nxt > last else last + 1
        return raw_start, raw_end

    def from_raw_span(self, start: int, end: int) -> tuple[int, int]:
        """Project a raw span onto the normalised text (smallest covering span)."""
        if not 0 <= start <= end <= len(self.raw):
            raise IndexError(f"span ({start}, {end}) outside raw text")
        om = self.offset_map
        n_start = next((i for i in range(len(self.text)) if om[i] >= start), len(self.text))
        n_end = next((i for i in range(n_start, len(self.text)) if om[i] >= end), len(self.text))
        return n_start, n_end


def _clusters(raw: str) -> list[tuple[int, str]]:
    """Split into (raw_start, base+combining marks) clusters for Unicode normalisation."""
    out: list[tuple[int, str]] = []
    i = 0
    while i < len(raw):
        j = i + 1
        while j < len(raw) and unicodedata.combining(raw[j]):
            j += 1
        out.append((i, raw[i:j]))
        i = j
    return out


def normalize(raw: str, config: NormalizationConfig | None = None) -> NormalizedText:
    """Normalise ``raw`` according to ``config`` while tracking offsets."""
    cfg = config or NormalizationConfig()
    ops: dict[str, int] = {}

    def bump(name: str, n: int = 1) -> None:
        ops[name] = ops.get(name, 0) + n

    # Stage 1: Unicode composition, cluster by cluster (keeps offsets monotone).
    chars: list[tuple[str, int]] = []
    for start, cluster in _clusters(raw):
        piece = cluster
        if cfg.fold_presentation_forms and any(_is_presentation_form(c) for c in piece):
            piece = unicodedata.normalize("NFKC", piece)
            bump("presentation_forms_folded")
        if cfg.unicode_form != "none":
            new = unicodedata.normalize(cfg.unicode_form, piece)
            if new != piece:
                bump(f"unicode_{cfg.unicode_form.lower()}")
            piece = new
        for k, ch in enumerate(piece):
            # Map each output char to the raw char it most plausibly came from.
            src = start + min(k, len(cluster) - 1) if len(piece) == len(cluster) else start
            chars.append((ch, src))

    # Stage 2: character-level deletions / substitutions.
    out: list[tuple[str, int]] = []
    for ch, src in chars:
        if cfg.remove_invisible_controls and ch in _INVISIBLE:
            bump("invisible_removed")
            continue
        if cfg.remove_tatweel and ch == TATWEEL:
            bump("tatweel_removed")
            continue
        if cfg.strip_diacritics and ch in _HARAKAT:
            bump("diacritics_removed")
            continue
        if cfg.strip_dagger_alef and ch == DAGGER_ALEF:
            bump("dagger_alef_removed")
            continue
        if cfg.strip_quranic_marks and ch in _QURANIC_MARKS:
            bump("quranic_marks_removed")
            continue
        if cfg.normalize_alef and ch in _ALEF_VARIANTS:
            ch = _ALEF_VARIANTS[ch]
            bump("alef_normalized")
        elif cfg.normalize_alef_maqsura and ch == "ى":
            ch = "ي"
            bump("alef_maqsura_normalized")
        elif cfg.normalize_teh_marbuta and ch == "ة":
            ch = "ه"
            bump("teh_marbuta_normalized")
        if cfg.digits == "western" and (ch in _ARABIC_INDIC or ch in _EXT_ARABIC_INDIC):
            ch = _ARABIC_INDIC.get(ch) or _EXT_ARABIC_INDIC[ch]
            bump("digits_converted")
        elif cfg.digits == "arabic_indic" and ch in _WESTERN_TO_ARABIC_INDIC:
            ch = _WESTERN_TO_ARABIC_INDIC[ch]
            bump("digits_converted")
        out.append((ch, src))

    # Stage 3: whitespace.
    if cfg.collapse_whitespace or cfg.max_consecutive_newlines is not None:
        ws: list[tuple[str, int]] = []
        for ch, src in out:
            if ch == "\r":
                bump("whitespace_normalized")
                continue
            if cfg.collapse_whitespace and ch in " \t   " :
                if ch != " ":
                    bump("whitespace_normalized")
                ch = " "
                if ws and ws[-1][0] == " ":
                    bump("whitespace_normalized")
                    continue
            if cfg.collapse_whitespace and ch == "\n":
                # strip trailing spaces on a line
                while ws and ws[-1][0] == " ":
                    ws.pop()
                    bump("whitespace_normalized")
            if ch == "\n" and cfg.max_consecutive_newlines is not None:
                run = 0
                for prev, _ in reversed(ws):
                    if prev == "\n":
                        run += 1
                    else:
                        break
                if run >= cfg.max_consecutive_newlines:
                    bump("whitespace_normalized")
                    continue
            ws.append((ch, src))
        out = ws
        if cfg.collapse_whitespace:
            while out and out[0][0] in " \n":
                out.pop(0)
            while out and out[-1][0] in " \n":
                out.pop()

    text = "".join(c for c, _ in out)
    offset_map = [s for _, s in out] + [len(raw)]
    return NormalizedText(text=text, raw=raw, offset_map=offset_map, operations=ops)


_WORD_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ\w]+", re.UNICODE)


def simple_tokenize(text: str) -> list[tuple[int, int, str]]:
    """Whitespace/punctuation tokenizer returning ``(start, end, token)``.

    Deliberately simple and dependency-free; morphological segmentation
    (clitic splitting) is out of scope for Phase 1 and must not be confused
    with the Stanford/Farasa segmentation used by the legacy scripts.
    """
    return [(m.start(), m.end(), m.group(0)) for m in _WORD_RE.finditer(text)]
