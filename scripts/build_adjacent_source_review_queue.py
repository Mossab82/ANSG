#!/usr/bin/env python3
"""Create a privacy-safe review queue for adjacent-category book sources."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


def read_jsonl(path: Path, compressed: bool) -> Iterator[dict[str, Any]]:
    opener = gzip.open if compressed else Path.open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, action="append", required=True,
                    help="All-source pretraining JSONL.GZ split; repeat for each split.")
    ap.add_argument("--sections", type=Path, required=True,
                    help="Heading-bound section candidates JSONL (metadata only).")
    ap.add_argument("--output", type=Path, required=True, help="UTF-8 CSV review queue.")
    ap.add_argument("--report", type=Path, required=True, help="JSON scope/count report.")
    args = ap.parse_args()

    books: dict[str, dict[str, Any]] = {}
    for path in args.corpus:
        for row in read_jsonl(path, compressed=True):
            if row.get("candidate_scope") != "narrative_adjacent":
                continue
            rid = str(row["record_id"])
            if rid in books:
                raise ValueError(f"Source record appears in more than one corpus split: {rid}")
            metadata = row.get("metadata", {})
            books[rid] = {
                "record_id": rid,
                "work_group": row.get("work_group"),
                "split": row.get("split"),
                "title": metadata.get("title"),
                "author": metadata.get("author"),
                "period": metadata.get("period"),
                "publisher_categories": " | ".join(metadata.get("publisher_categories", [])),
                "translation": metadata.get("translation"),
                "source_text_sha256": row.get("source_text_sha256"),
                "heading_sections": 0,
                "heading_section_sentencepiece_pieces": 0,
                "review_status": "pending_human_content_and_boundary_review",
            }

    for row in read_jsonl(args.sections, compressed=False):
        rid = str(row["source_record_id"])
        if rid not in books:
            continue
        book = books[rid]
        if row.get("candidate_scope") != "narrative_adjacent":
            continue
        if row.get("work_group") != book["work_group"] or row.get("split") != book["split"]:
            raise ValueError(f"Section provenance/split mismatch for {row.get('candidate_id')}")
        if row.get("source_text_sha256") != book["source_text_sha256"]:
            raise ValueError(f"Section source hash mismatch for {row.get('candidate_id')}")
        book["heading_sections"] += 1
        book["heading_section_sentencepiece_pieces"] += int(row["direct_sentencepiece_tokens"])

    rows = sorted(books.values(), key=lambda x: (-x["heading_section_sentencepiece_pieces"],
                                                  str(x["title"] or ""), x["record_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["record_id", "work_group", "split", "title", "author",
                                         "period", "publisher_categories", "translation",
                                         "source_text_sha256", "heading_sections",
                                         "heading_section_sentencepiece_pieces", "review_status"]
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "schema": "ansg.adjacent_source_review_queue_report/1",
        "status": "metadata_only_review_queue_not_story_classification",
        "adjacent_source_records": len(rows),
        "records_with_heading_sections": sum(bool(row["heading_sections"]) for row in rows),
        "heading_sections": sum(row["heading_sections"] for row in rows),
        "heading_section_sentencepiece_pieces": sum(
            row["heading_section_sentencepiece_pieces"] for row in rows
        ),
        "human_review_completed": False,
        "annotations_created": 0,
        "source_text_embedded": False,
        "limitations": [
            "Adjacent-category source titles/categories do not prove that the text is a story.",
            "Heading-bounded pieces are provisional source spans, not verified narrative content.",
            "Every row requires a human content/relevance review before adding it to a story corpus.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
