#!/usr/bin/env python3
"""Index reported-tale opening formulas in the 1816 Kalila wa-Dimna text.

The output records opening cues and source offsets only. It deliberately does
not infer tale endings, nesting depth, gold story boundaries, or annotations.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


MARKS = r"[\u064b-\u065f\u0670\u0640]*"
REPORTED_TALE_INTRO = re.compile(
    rf"(?P<cue>زعم(?:وا)?{MARKS}\s*(?:ا|أ|إ|آ){MARKS}ن{MARKS}(?:ه{MARKS})?)"
)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSONL records from the pinned Kalila 1816 Wikisource edition.")
    parser.add_argument("--output", type=Path, required=True,
                        help="JSONL marker inventory; no source text is copied.")
    parser.add_argument("--report", type=Path, required=True, help="JSON count report.")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    chapters: Counter[str] = Counter()
    source_ids: set[str] = set()
    for record in read_jsonl(args.input):
        source_id = str(record["record_id"])
        source_ids.add(source_id)
        text = str(record.get("text", ""))
        source = record.get("source", {})
        chapter = (record.get("narrative", {}).get("chapter_title") or
                   source.get("page_title") or source_id)
        for index, match in enumerate(REPORTED_TALE_INTRO.finditer(text), 1):
            rows.append({
                "schema": "ansg.kalila_reported_tale_opening/1",
                "marker_id": f"{source_id}-reported-tale-{index:03d}",
                "source_record_id": source_id,
                "source_chapter_title": chapter,
                "source_text_sha256": record.get("source_text_sha256"),
                "opening_start_char": match.start(),
                "opening_end_char": match.end(),
                "opening_cue": match.group("cue"),
                "parent_source_chapter_id": source_id,
                "story_end_char": None,
                "nesting_depth": None,
                "split": "unassigned",
                "status": "reported_tale_opening_candidate_requires_boundary_review",
                "annotations": {"DIA": None, "MOTIF": None, "MORAL": None},
                "review_flags": [
                    "confirm_marker_introduces_a_narrative_not_a_reference",
                    "locate_tale_end_and_parent_resumption_from_source_context",
                    "do_not_assume_one_cue_equals_one_complete_story",
                ],
            })
            chapters[source_id] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    report = {
        "schema": "ansg.kalila_reported_tale_opening_report/1",
        "status": "opening_cues_only_not_story_segmentation",
        "edition": "Kalila wa-Dimna, Paris 1816; pinned Wikisource source revisions",
        "source_chapters": len(source_ids),
        "reported_tale_opening_candidates": len(rows),
        "candidates_by_source_chapter_id": dict(sorted(chapters.items())),
        "source_text_embedded": False,
        "story_endings_inferred": False,
        "nesting_relations_inferred": False,
        "human_annotations_added": 0,
        "triple_covered_tokens": 0,
        "limitations": [
            "The formula can introduce a nested tale, an illustrative parable, or a cited example.",
            "An opening marker does not identify the embedded tale's ending or the parent's resumption.",
            "Chapter context and source-text review are required before treating a marker as a story unit.",
            "All source chapters remain in one direct-build split group.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
