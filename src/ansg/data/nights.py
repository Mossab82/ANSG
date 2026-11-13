"""Ingestion of *One Thousand and One Nights* (ألف ليلة وليلة) source files.

No corpus text is bundled with this repository. The pipeline reads the
licensed edition files you supply and converts them into ANHC
:class:`~ansg.data.records.NarrativeRecord` objects.

Structure handled
-----------------
The Nights have two cross-cutting divisions:

* **Nights** — the frame-tale division marked by night headings
  ("الليلة الأولى", …). Detected with ``night_heading_pattern``.
* **Tales** — the story division marked by tale titles ("حكاية …").
  Detected with ``story_title_pattern``.

Each tale becomes one record; the record keeps the list of nights it spans.
Text that precedes the first tale title (e.g. the frame prologue) becomes a
record with ``story_index = 0`` when ``keep_preamble`` is true.

Rhetorical formulae of the frame narrative (e.g. "بلغني أيها الملك السعيد",
"وأدرك شهرزاد الصباح فسكتت عن الكلام المباح") are *detected and recorded* as
metadata with offsets — never stripped — so that later stages can choose to
model or mask them explicitly.

Edition-specific heading conventions differ; all patterns are configurable
and the defaults are documented as defaults, not as properties of any
particular edition used in the dissertation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..utils.provenance import config_hash, sha256_file, sha256_text
from .arabic_numbers import parse_arabic_number
from .normalization import NormalizationConfig, normalize
from .records import NarrativeMetadata, NarrativeRecord, SourceProvenance, Span

DEFAULT_NIGHT_HEADING = r"^[ \t]*(?:و[ \t]*)?(?:في[ \t]+)?(?:ال)?ليل[ةه][ \t]+(?P<num>[^\n]{1,60}?)[ \t]*$"
DEFAULT_STORY_TITLE = r"^[ \t]*(?P<title>(?:حكاية|قصة|حديث)[ \t]+[^\n]{2,120}?)[ \t]*$"
DEFAULT_FORMULAE: dict[str, str] = {
    "night_opening": r"بلغني\s+أيها\s+الملك\s+السعيد",
    "night_closing": r"(?:و)?أدرك\s+شهرزاد\s+الصباح\s+فسكتت\s+عن\s+الكلام\s+المباح",
}


@dataclass(frozen=True)
class NightsConfig:
    collection: str = "alf_layla_wa_layla"
    night_heading_pattern: str = DEFAULT_NIGHT_HEADING
    story_title_pattern: str = DEFAULT_STORY_TITLE
    formulae: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_FORMULAE))
    keep_preamble: bool = True
    keep_headings_in_text: bool = True  # night headings are part of the narrative text
    min_story_chars: int = 1
    encoding: str = "utf-8"

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any] | None) -> "NightsConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown nights options: {sorted(unknown)}")
        return cls(**d)


@dataclass
class NightMarker:
    start: int
    end: int
    heading: str
    number: int | None


@dataclass
class StoryUnit:
    index: int
    title: str | None
    start: int  # offsets in the normalised source text
    end: int
    nights: list[int]
    unparsed_night_headings: list[str]
    formulae: list[dict[str, Any]]  # offsets relative to text[start:end]


@dataclass
class IngestionReport:
    source_file: str
    file_sha256: str
    n_chars: int
    n_night_headings: int
    n_unparsed_night_headings: int
    n_stories: int
    night_number_gaps: list[tuple[int, int]]
    night_number_regressions: list[tuple[int, int]]
    warnings: list[str]


def find_night_markers(text: str, cfg: NightsConfig) -> list[NightMarker]:
    rx = re.compile(cfg.night_heading_pattern, re.MULTILINE)
    out = []
    for m in rx.finditer(text):
        num_txt = m.groupdict().get("num") or ""
        out.append(NightMarker(m.start(), m.end(), m.group(0).strip(), parse_arabic_number(num_txt)))
    return out


def find_story_titles(text: str, cfg: NightsConfig) -> list[tuple[int, int, str]]:
    rx = re.compile(cfg.story_title_pattern, re.MULTILINE)
    return [(m.start(), m.end(), (m.groupdict().get("title") or m.group(0)).strip()) for m in rx.finditer(text)]


def find_formulae(text: str, cfg: NightsConfig, start: int = 0, end: int | None = None) -> list[dict[str, Any]]:
    end = len(text) if end is None else end
    found: list[dict[str, Any]] = []
    for name, pattern in cfg.formulae.items():
        for m in re.finditer(pattern, text[start:end]):
            found.append({"type": name, "start": m.start(), "end": m.end(), "text": m.group(0)})
    found.sort(key=lambda f: (f["start"], f["type"]))
    return found


def segment_stories(text: str, cfg: NightsConfig) -> tuple[list[StoryUnit], list[NightMarker]]:
    """Split normalised source text into tale units with night membership."""
    markers = find_night_markers(text, cfg)
    titles = find_story_titles(text, cfg)
    bounds: list[tuple[int, str | None]] = []
    if cfg.keep_preamble and (not titles or titles[0][0] > 0):
        bounds.append((0, None))
    bounds.extend((s, t) for s, _, t in titles)
    units: list[StoryUnit] = []
    for i, (start, title) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else len(text)
        seg = text[start:end]
        if len(seg.strip()) < cfg.min_story_chars:
            continue
        # A tale belongs to the night in progress at its start plus every night begun inside it.
        # Unparsed headings are reported but never used to assign night membership.
        prior = [m for m in markers if m.start < start and m.number is not None]
        inside = [m for m in markers if start <= m.start < end]
        relevant = ([prior[-1]] if prior else []) + inside
        nights = [m.number for m in relevant if m.number is not None]
        unparsed = [m.heading for m in inside if m.number is None]
        forms = find_formulae(text, cfg, start, end)
        units.append(StoryUnit(
            index=len(units),
            title=title, start=start, end=end, nights=sorted(set(nights)),
            unparsed_night_headings=unparsed, formulae=forms,
        ))
    return units, markers


def _night_sequence_issues(markers: Sequence[NightMarker]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    nums = [m.number for m in markers if m.number is not None]
    gaps, regressions = [], []
    for a, b in zip(nums, nums[1:]):
        if b > a + 1:
            gaps.append((a, b))
        elif b <= a:
            regressions.append((a, b))
    return gaps, regressions


def ingest_file(
    path: str | Path,
    provenance: SourceProvenance,
    nights_cfg: NightsConfig | None = None,
    norm_cfg: NormalizationConfig | None = None,
) -> tuple[list[NarrativeRecord], IngestionReport]:
    """Read one source file and return ANHC records plus an ingestion report."""
    cfg = nights_cfg or NightsConfig()
    ncfg = norm_cfg or NormalizationConfig()
    path = Path(path)
    raw = path.read_bytes().decode(cfg.encoding)
    norm = normalize(raw, ncfg)
    text = norm.text
    units, markers = segment_stories(text, cfg)

    file_hash = sha256_file(path)
    prov = SourceProvenance(**{**provenance.__dict__, "file_path": path.name, "file_sha256": file_hash})
    ncfg_hash = config_hash(ncfg.to_dict())

    records: list[NarrativeRecord] = []
    warnings: list[str] = []
    for u in units:
        body = text[u.start:u.end]
        if not cfg.keep_headings_in_text:
            body = re.sub(cfg.night_heading_pattern, "", body, flags=re.MULTILINE)
        body = body.strip()
        # Stable identifier: source + position + content hash (order and content dependent by design).
        rid = f"{prov.source_id}:{u.index:05d}:{sha256_text(body)[:12]}"
        forms = find_formulae(body, cfg)  # offsets relative to the record text
        raw_s, raw_e = norm.to_raw_span(u.start, u.end)
        meta = NarrativeMetadata(
            collection=cfg.collection,
            title=u.title,
            story_index=u.index,
            night_numbers=u.nights,
            source_span=Span(u.start, u.end),
            formulae=forms,
            extra={"raw_source_span": [raw_s, raw_e],
                   "unparsed_night_headings": u.unparsed_night_headings},
        )
        if u.unparsed_night_headings:
            warnings.append(f"{rid}: {len(u.unparsed_night_headings)} night heading(s) not parsed")
        records.append(NarrativeRecord(
            record_id=rid, text=body, source=prov, narrative=meta,
            normalization_config_sha256=ncfg_hash,
        ))

    gaps, regressions = _night_sequence_issues(markers)
    if not markers:
        warnings.append("no night headings detected; check night_heading_pattern for this edition")
    if not any(u.title for u in units):
        warnings.append("no tale titles detected; check story_title_pattern for this edition")
    report = IngestionReport(
        source_file=str(path), file_sha256=file_hash, n_chars=len(text),
        n_night_headings=len(markers),
        n_unparsed_night_headings=sum(m.number is None for m in markers),
        n_stories=len(records), night_number_gaps=gaps,
        night_number_regressions=regressions, warnings=warnings,
    )
    return records, report
