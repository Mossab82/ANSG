#!/usr/bin/env python3
"""Rank core book records for human story-content review using title cues only.

This is a metadata triage aid, not a narrative classifier. Every core source is
included, title cues only set review priority, and no source prose or labels are
added.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


TITLE_CUES = {
    # Patterns target the normalized forms produced by normalize().
    "story_or_tale": re.compile(r"(?:قص(?:ه|ص|صص)|حكاي(?:ه|ات)|رواي(?:ه|ات)|الف\s*ليله|ليالي|نوادر|سيره|حكايات شعبيه|حكايه شعبيه)", re.I),
    "translated_tales_or_novels": re.compile(r"(?:stories|story|tales|tale|folktales|folktale|novel|short fiction)", re.I),
}
FIELDS = [
    "priority_rank", "title_signal_priority", "title_signal_terms", "source_record_id",
    "title", "author", "original_title", "original_author", "publisher_categories",
    "period", "translation", "license", "source_text_sha256", "text_file",
    "observed_whitespace_words", "sentencepiece_tokens", "heading_section_count",
    "heading_section_sentencepiece_tokens", "direct_build_split", "work_group",
    "review_status", "content_is_narrative", "story_boundary_status", "reviewed_title",
    "reviewer_id", "review_evidence", "reviewer_notes",
]


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"[ً-ْٰـ]", "", s)
    s = re.sub(r"[إأآٱ]", "ا", s).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True, help="Direct-build source records JSONL")
    parser.add_argument("--sections", type=Path, required=True, help="Heading-section candidates JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Metadata-only CSV queue")
    parser.add_argument("--report", type=Path, required=True, help="JSON summary")
    args = parser.parse_args()

    sections_by_id: dict[str, list[dict[str, Any]]] = {}
    for section in read_jsonl(args.sections):
        sections_by_id.setdefault(str(section["source_record_id"]), []).append(section)

    rows: list[dict[str, Any]] = []
    cue_counts: Counter[str] = Counter()
    for source in read_jsonl(args.sources):
        if source.get("candidate_scope") != "narrative_core":
            continue
        title = str(source.get("title", ""))
        original_title = str(source.get("original_title", ""))
        combined_title = f"{title} {original_title}"
        normalized = normalize(combined_title)
        matched = [name for name, pattern in TITLE_CUES.items() if pattern.search(normalized)]
        cue_counts.update(matched)
        source_sections = sections_by_id.get(str(source["record_id"]), [])
        rows.append({
            "title_signal_priority": "title_cue_match" if matched else "no_title_cue",
            "title_signal_terms": ";".join(matched),
            "source_record_id": source["record_id"],
            "title": title,
            "author": source.get("author", ""),
            "original_title": original_title,
            "original_author": source.get("original_author", ""),
            "publisher_categories": ";".join(source.get("publisher_categories", [])),
            "period": source.get("period", ""),
            "translation": source.get("translation", ""),
            "license": source.get("license", ""),
            "source_text_sha256": source.get("source_text_sha256", ""),
            "text_file": source.get("text_file", ""),
            "observed_whitespace_words": source.get("observed_whitespace_words", 0),
            "sentencepiece_tokens": source.get("sentencepiece_tokens", 0),
            "heading_section_count": len(source_sections),
            "heading_section_sentencepiece_tokens": sum(int(s.get("direct_sentencepiece_tokens", 0)) for s in source_sections),
            "direct_build_split": source.get("split", "unassigned"),
            "work_group": source.get("work_group", ""),
            "review_status": "pending_human_content_review",
            "content_is_narrative": "",
            "story_boundary_status": "",
            "reviewed_title": "",
            "reviewer_id": "",
            "review_evidence": "",
            "reviewer_notes": "",
            "_signal_rank": 0 if matched else 1,
            "_size_rank": -int(source.get("sentencepiece_tokens", 0)),
        })

    rows.sort(key=lambda row: (row["_signal_rank"], row["_size_rank"], normalize(row["title"]), row["source_record_id"]))
    for rank, row in enumerate(rows, 1):
        row["priority_rank"] = rank
        row.pop("_signal_rank", None)
        row.pop("_size_rank", None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    matched_count = sum(1 for row in rows if row["title_signal_priority"] == "title_cue_match")
    report = {
        "schema": "ansg.core_story_source_review_queue_report/1",
        "status": "title_metadata_triage_not_story_classification",
        "source_records": len(rows),
        "title_cue_matched_records": matched_count,
        "records_without_title_cues": len(rows) - matched_count,
        "heading_sections_across_core_sources": sum(int(row["heading_section_count"]) for row in rows),
        "sentencepiece_tokens_across_core_sources": sum(int(row["sentencepiece_tokens"]) for row in rows),
        "title_cue_counts_nonexclusive": dict(sorted(cue_counts.items())),
        "source_text_embedded": False,
        "content_classifications_prefilled": False,
        "annotations_prefilled": False,
        "priority_definition": "Rows with literal story/tale terms in Arabic or English title fields sort first; ties sort by descending direct-build SentencePiece count.",
        "limitations": [
            "A title cue does not establish that the book contains stories; human content review is required.",
            "A book without a title cue can still contain stories and remains in the queue.",
            "Source token counts are direct-build SentencePiece counts, not the thesis tokenizer's counts.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
