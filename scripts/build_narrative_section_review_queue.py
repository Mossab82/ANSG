#!/usr/bin/env python3
"""Create a metadata-only review queue for core/classical heading sections.

Heading spans are retrieval candidates, not verified story units. This queue
contains provenance and review fields only; it never copies source prose or
prepopulates annotation labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


SCOPES = {"narrative_core", "classical_narrative"}
FIELDS = [
    "candidate_id", "source_record_id", "source_title", "source_author",
    "candidate_scope", "source_text_sha256", "text_file", "license",
    "publisher_categories", "period", "translation", "heading_text",
    "boundary_cue", "start_char", "end_char", "heading_line",
    "observed_whitespace_words", "direct_sentencepiece_tokens", "work_group",
    "direct_build_split", "review_status", "content_is_narrative",
    "is_complete_story_unit", "boundary_decision", "reviewed_title",
    "parent_story_id", "nesting_depth", "DIA", "MOTIF", "MORAL",
    "reviewer_id", "review_evidence", "reviewer_notes",
]


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
    parser.add_argument("--sections", type=Path, required=True,
                        help="JSONL from segment_story_boundary_candidates.py")
    parser.add_argument("--sources", type=Path, required=True,
                        help="Source-register JSONL keyed by record_id")
    parser.add_argument("--output", type=Path, required=True,
                        help="CSV filename prefix; output is split into source-preserving shards")
    parser.add_argument("--rows-per-shard", type=int, default=1000,
                        help="Maximum rows per CSV shard; a source record is never split across shards")
    parser.add_argument("--report", type=Path, required=True, help="JSON queue summary")
    args = parser.parse_args()

    sources: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(args.sources):
        rid = str(record["record_id"])
        if rid in sources:
            raise ValueError(f"Duplicate source record id: {rid}")
        sources[rid] = record

    rows: list[dict[str, Any]] = []
    scope_rows: Counter[str] = Counter()
    scope_tokens: Counter[str] = Counter()
    scope_records: dict[str, set[str]] = defaultdict(set)
    missing_source_ids: set[str] = set()
    seen_candidate_ids: set[str] = set()
    for section in read_jsonl(args.sections):
        scope = str(section.get("candidate_scope", "unspecified"))
        if scope not in SCOPES:
            continue
        candidate_id = str(section["candidate_id"])
        if candidate_id in seen_candidate_ids:
            raise ValueError(f"Duplicate section candidate id: {candidate_id}")
        seen_candidate_ids.add(candidate_id)
        rid = str(section["source_record_id"])
        source = sources.get(rid)
        if source is None:
            missing_source_ids.add(rid)
            source = {}
        if source and source.get("source_text_sha256") != section.get("source_text_sha256"):
            raise ValueError(f"Source hash mismatch for {candidate_id}")
        if source and source.get("candidate_scope") != scope:
            raise ValueError(f"Candidate scope mismatch for {candidate_id}")

        rows.append({
            "candidate_id": candidate_id,
            "source_record_id": rid,
            "source_title": source.get("title", section.get("source_title", "")),
            "source_author": source.get("author", ""),
            "candidate_scope": scope,
            "source_text_sha256": section.get("source_text_sha256", ""),
            "text_file": source.get("text_file", ""),
            "license": source.get("license", ""),
            "publisher_categories": ";".join(source.get("publisher_categories", [])),
            "period": source.get("period", ""),
            "translation": source.get("translation", ""),
            "heading_text": section.get("heading_text", ""),
            "boundary_cue": section.get("boundary_cue", ""),
            "start_char": section.get("start_char", ""),
            "end_char": section.get("end_char", ""),
            "heading_line": section.get("heading_line", ""),
            "observed_whitespace_words": section.get("observed_whitespace_words", 0),
            "direct_sentencepiece_tokens": section.get("direct_sentencepiece_tokens", 0),
            "work_group": section.get("work_group", ""),
            "direct_build_split": section.get("split", "unassigned"),
            "review_status": "pending_human_content_and_boundary_review",
            "content_is_narrative": "",
            "is_complete_story_unit": "",
            "boundary_decision": "",
            "reviewed_title": "",
            "parent_story_id": "",
            "nesting_depth": "",
            "DIA": "",
            "MOTIF": "",
            "MORAL": "",
            "reviewer_id": "",
            "review_evidence": "",
            "reviewer_notes": "",
        })
        scope_rows[scope] += 1
        scope_tokens[scope] += int(section.get("direct_sentencepiece_tokens", 0))
        scope_records[scope].add(rid)

    if args.rows_per_shard < 1:
        parser.error("--rows-per-shard must be positive")
    rows.sort(key=lambda row: (row["candidate_scope"], row["source_record_id"], row["start_char"]))
    rows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_source[row["source_record_id"]].append(row)
    shards: list[list[dict[str, Any]]] = []
    shard: list[dict[str, Any]] = []
    for source_id in sorted(rows_by_source):
        source_rows = rows_by_source[source_id]
        if shard and len(shard) + len(source_rows) > args.rows_per_shard:
            shards.append(shard)
            shard = []
        shard.extend(source_rows)
    if shard:
        shards.append(shard)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_shards: list[dict[str, Any]] = []
    for shard_index, shard_rows in enumerate(shards, 1):
        shard_path = args.output.with_name(f"{args.output.stem}-{shard_index:03d}{args.output.suffix or '.csv'}")
        with shard_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(shard_rows)
        output_shards.append({"path": shard_path.name, "rows": len(shard_rows)})

    report = {
        "schema": "ansg.narrative_section_review_queue_report/1",
        "status": "metadata_only_heading_section_review_queue",
        "candidate_scopes": sorted(scope_rows),
        "candidate_rows": len(rows),
        "source_records": len({row["source_record_id"] for row in rows}),
        "rows_by_scope": dict(sorted(scope_rows.items())),
        "source_records_by_scope": {scope: len(ids) for scope, ids in sorted(scope_records.items())},
        "direct_sentencepiece_tokens_by_scope": dict(sorted(scope_tokens.items())),
        "missing_source_records": sorted(missing_source_ids),
        "source_text_embedded": False,
        "source_records_are_kept_within_one_shard": True,
        "rows_per_shard_target": args.rows_per_shard,
        "shards": output_shards,
        "headings_are_verified_story_boundaries": False,
        "annotation_fields_prefilled": False,
        "direct_build_split_is_thesis_split": False,
        "review_policy": [
            "Resolve each source file using the source record id and verify its SHA-256 before using offsets.",
            "Review content and both span boundaries; a chapter or section heading may introduce non-narrative material.",
            "Review duplicates, translations, and larger story cycles before assigning a final group-level split.",
            "Populate DIA, MOTIF, and MORAL only from documented evidence under the annotation codebook.",
        ],
        "limitations": [
            "This is a navigation and review queue, not a story corpus or gold segmentation.",
            "SentencePiece counts are direct-build counts and do not reproduce the thesis tokenizer count.",
            "The existing split is a direct-build provisional split, not the thesis TF-IDF/DBSCAN stratified split.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
