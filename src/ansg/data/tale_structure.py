"""Tale boundaries and frame/embedded-tale relations from ordered text blocks. Torch-free.

Input: the ordered :class:`~ansg.data.mediawiki.Block` sequence of an edition
(headings with their levels, and text blocks), e.g. from ``EditionFetcher``.
Output: a list of :class:`TaleNode` with spans, night numbers, an evidence-backed
``parent`` (if any) and recorded ambiguities; then ANHC ``NarrativeRecord``s whose
``parent_story_id`` is set only from accepted evidence.

Evidence policy (nothing is invented)
-------------------------------------
* ``heading_hierarchy`` — a tale heading nested under an open tale heading of a
  *higher* level (h3 under h2). Accepted by default.
* ``textual_cue`` — a tale title appears while a parent tale is open and the
  block just before it contains a configured framing cue (a character announcing
  a story). Recorded as a candidate; used for ``parent_story_id`` only when
  ``accept_textual_cues=True``.
* Anything else (titles at the same heading level, titles in running text
  without a cue) is recorded as an **ambiguity**, never as a relation.

Known limit: without explicit markup the point where a parent tale *resumes*
after an embedded tale cannot be located; text after an embedded tale stays with
the last open tale and an ambiguity is logged. Splitting by story cycle
(:func:`cycle_split`) keeps such text on the same side of the split anyway.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .arabic_numbers import parse_arabic_number
from .leakage import story_cycle_groups
from .records import NarrativeMetadata, NarrativeRecord, SourceProvenance
from .splits import SplitConfig, assign_splits, check_no_leakage

DEFAULT_TITLE_PATTERN = r"^\s*(?:ال)?(?:حكاية|حكايات|قصة|حديث)\s+\S+"
"""Matches 'حكاية …', the plural 'حكايات …' (e.g. the frame tale of Shahriyar) and 'الحكاية الثانية …'."""
DEFAULT_NIGHT_PATTERN = r"(?:فلما\s+)?(?:كانت\s+)?الليلة\s+(?P<num>[ء-ي\s]+?|\d+)\s*$"
DEFAULT_FRAMING_CUES = ("وما حكاية", "حكايتي", "أحكي لك", "أحكي لكم", "أحدثك", "سأحدثك", "اسمع حكاية",
                        "فحكى", "فحكت", "فقص عليه")


@dataclass
class StructureConfig:
    title_pattern: str = DEFAULT_TITLE_PATTERN
    night_pattern: str = DEFAULT_NIGHT_PATTERN
    framing_cues: tuple[str, ...] = DEFAULT_FRAMING_CUES
    title_in_text_blocks: bool = True       # also recognise short text blocks that look like titles
    max_title_chars: int = 80
    accept_textual_cues: bool = False


@dataclass
class TaleNode:
    id: str
    title: str
    start_block: int
    end_block: int                          # exclusive
    heading_level: int | None               # None = title found in a text block
    parent: str | None = None
    nesting_evidence: str | None = None     # heading_hierarchy | textual_cue
    candidate_parent: str | None = None     # textual-cue candidate when not accepted
    nights: list[int] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)


@dataclass
class StructureReport:
    tales: list[TaleNode]
    night_blocks: dict[int, int]            # block index -> night number
    ambiguities: list[str]
    unparsed_night_headings: list[str]


def _night_number(text: str, pattern: re.Pattern) -> tuple[bool, int | None]:
    m = pattern.search(text.strip())
    if not m or len(text) > 60:
        return False, None
    return True, parse_arabic_number(m.group("num"))


def analyse_structure(blocks: Sequence, config: StructureConfig | None = None) -> StructureReport:
    cfg = config or StructureConfig()
    title_re = re.compile(cfg.title_pattern)
    night_re = re.compile(cfg.night_pattern)
    nights: dict[int, int] = {}
    unparsed: list[str] = []
    titles: list[tuple[int, str, int | None]] = []
    for i, b in enumerate(blocks):
        is_night, num = _night_number(b.text, night_re)
        if is_night:
            if num is None:
                unparsed.append(b.text)
            else:
                nights[i] = num
            continue
        if b.kind == "heading" and title_re.search(b.text):
            titles.append((i, b.text.strip(), b.level))
        elif cfg.title_in_text_blocks and b.kind == "text" and len(b.text) <= cfg.max_title_chars \
                and title_re.search(b.text) and not b.text.strip().endswith((".", ":", "؟", "!")):
            titles.append((i, b.text.strip(), None))

    tales: list[TaleNode] = []
    ambiguities: list[str] = []
    open_stack: list[TaleNode] = []          # open tales by heading level
    for n, (i, title, level) in enumerate(titles):
        end = titles[n + 1][0] if n + 1 < len(titles) else len(blocks)
        node = TaleNode(id=f"tale-{n + 1:04d}", title=title, start_block=i, end_block=end, heading_level=level)
        if level is not None:
            while open_stack and (open_stack[-1].heading_level or 99) >= level:
                open_stack.pop()
            if open_stack:
                node.parent, node.nesting_evidence = open_stack[-1].id, "heading_hierarchy"
            open_stack.append(node)
        prev_text = blocks[i - 1].text if i > 0 else ""
        cue = next((c for c in cfg.framing_cues if c in prev_text), None)
        if node.parent is None and tales:
            if cue:
                cand = (open_stack[-2] if level is not None and len(open_stack) > 1 else tales[-1])
                if level is not None and cand is node:
                    cand = tales[-1]
                node.candidate_parent = cand.id
                msg = f"{node.id} '{title}': framing cue {cue!r} in the preceding block suggests it is " \
                      f"told inside {cand.id} '{cand.title}'"
                if cfg.accept_textual_cues:
                    node.parent, node.nesting_evidence = cand.id, "textual_cue"
                else:
                    node.ambiguities.append(msg + " (candidate only; not accepted)")
            elif level is None or (tales[-1].heading_level == level):
                node.ambiguities.append(f"{node.id} '{title}': no markup or cue decides whether it is embedded "
                                        f"in {tales[-1].id} or a sibling; recorded as top-level")
        if node.parent is not None:
            ambiguities.append(f"{node.id}: resumption point of parent {node.parent} after this tale is not "
                               "marked; following text stays with the last open tale")
        tales.append(node)
    # nights covered: the night in progress at the tale start plus every night started inside it
    ordered = sorted(nights.items())
    for t in tales:
        prior = [num for bi, num in ordered if bi < t.start_block]
        inside = [num for bi, num in ordered if t.start_block <= bi < t.end_block]
        t.nights = sorted(set(([prior[-1]] if prior else []) + inside))
        ambiguities += t.ambiguities
    if not titles:
        ambiguities.append("no tale titles recognised; check title_pattern for this edition")
    return StructureReport(tales, nights, ambiguities, unparsed)


def tale_records(blocks: Sequence, report: StructureReport, provenance: SourceProvenance,
                 collection: str = "alf_layla") -> list[NarrativeRecord]:
    """One ANHC record per tale; ``parent_story_id`` only from accepted evidence."""
    ids = {t.id: f"{provenance.source_id}:{t.id}" for t in report.tales}
    records = []
    for t in report.tales:
        text = "\n\n".join(b.text for b in blocks[t.start_block:t.end_block])
        meta = NarrativeMetadata(
            collection=collection, title=t.title, story_index=int(t.id.split("-")[1]),
            night_numbers=t.nights, parent_story_id=ids.get(t.parent) if t.parent else None,
            extra={"nesting_evidence": t.nesting_evidence, "candidate_parent": ids.get(t.candidate_parent),
                   "ambiguities": t.ambiguities, "block_span": [t.start_block, t.end_block]})
        records.append(NarrativeRecord(record_id=ids[t.id], text=text, source=provenance, narrative=meta))
    return records


def cycle_split(records: Sequence[NarrativeRecord], config: SplitConfig) -> dict:
    """Split by story cycle (frame tale + every tale embedded in it) and audit the result."""
    roots = story_cycle_groups(records)
    assignment = assign_splits(records, config, group_fn=lambda r: roots[r.record_id])
    leaks = check_no_leakage(records, lambda r: roots[r.record_id])
    return {"assignment": assignment, "cycles": len(set(roots.values())), "leaking_cycles": leaks}


def unresolved(report: StructureReport) -> Iterable[str]:
    return list(report.ambiguities) + [f"unparsed night heading: {h!r}" for h in report.unparsed_night_headings]
