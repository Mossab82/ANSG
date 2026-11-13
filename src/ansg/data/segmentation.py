"""Paragraph and sentence segmentation with character offsets.

Segmentation is rule-based and deterministic. Sentence boundaries default to
the full stop, Arabic question mark (؟), exclamation mark, Arabic semicolon
(؛) and ellipsis. The legacy ``Sentances_extractor`` split on the Arabic comma
(،); that is available via ``split_on_arabic_comma=True`` but disabled by
default because in classical prose the comma frequently separates clauses of a
single rhymed period rather than sentences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    text: str
    kind: str  # "paragraph" | "sentence"
    index: int


@dataclass(frozen=True)
class SegmentationConfig:
    paragraph_separator: str = r"\n\s*\n"  # blank line
    single_newline_is_paragraph: bool = False
    sentence_terminators: str = ".!?؟؛…"
    split_on_arabic_comma: bool = False
    min_sentence_chars: int = 1

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "SegmentationConfig":
        data = dict(data or {})
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown segmentation options: {sorted(unknown)}")
        return cls(**data)


def _strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def segment_paragraphs(text: str, config: SegmentationConfig | None = None) -> list[Segment]:
    cfg = config or SegmentationConfig()
    pattern = r"\n" if cfg.single_newline_is_paragraph else cfg.paragraph_separator
    segs: list[Segment] = []
    pos = 0
    for m in list(re.finditer(pattern, text)) + [None]:
        end = m.start() if m else len(text)
        s, e = _strip_span(text, pos, end)
        if e > s:
            segs.append(Segment(s, e, text[s:e], "paragraph", len(segs)))
        if m:
            pos = m.end()
    return segs


def segment_sentences(
    text: str, config: SegmentationConfig | None = None, offset: int = 0
) -> list[Segment]:
    """Split ``text`` into sentences; offsets are relative to ``text`` + ``offset``."""
    cfg = config or SegmentationConfig()
    terms = cfg.sentence_terminators + ("،" if cfg.split_on_arabic_comma else "")
    boundary = re.compile("[" + re.escape(terms) + "]+")
    segs: list[Segment] = []
    for para in segment_paragraphs(text, cfg):
        pos = para.start
        for m in boundary.finditer(text, para.start, para.end):
            # A full stop between digits (e.g. 3.5) is not a sentence boundary.
            if (
                m.group(0) == "."
                and m.start() > 0
                and text[m.start() - 1].isdigit()
                and m.end() < len(text)
                and text[m.end()].isdigit()
            ):
                continue
            # include closing quotes/brackets directly after the terminator
            end = m.end()
            while end < para.end and text[end] in "\"'»”)]}":
                end += 1
            s, e = _strip_span(text, pos, end)
            if e - s >= cfg.min_sentence_chars:
                segs.append(Segment(s + offset, e + offset, text[s:e], "sentence", len(segs)))
            pos = end
        s, e = _strip_span(text, pos, para.end)
        if e - s >= cfg.min_sentence_chars:
            segs.append(Segment(s + offset, e + offset, text[s:e], "sentence", len(segs)))
    return segs
