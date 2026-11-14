#!/usr/bin/env python3
"""Extract inventoried Alf Layla spans from the licensed e-book corpus.

The resulting JSONL contains text and provenance for review or prototype use.
Each span remains a candidate: the boundary is the next structural heading,
not a human-verified story end. No annotation labels are generated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Iterator


DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u0640]")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc


def normalize_heading(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = DIACRITICS.sub("", value).strip()
    return value.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True,
                        help="Provisional story unit inventory JSONL")
    parser.add_argument("--sources", type=Path, required=True,
                        help="Direct-build source register JSONL")
    parser.add_argument("--sections", type=Path, required=True,
                        help="Section candidates with direct-build SentencePiece counts")
    parser.add_argument("--source-zip", type=Path, required=True,
                        help="Original Arabic E-Book Corpus text ZIP")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for text JSONL shards")
    parser.add_argument("--max-bytes-per-file", type=int, default=700_000,
                        help="Target UTF-8 size per JSONL shard; a record is never split")
    parser.add_argument("--report", type=Path, required=True, help="JSON summary")
    args = parser.parse_args()

    source_by_id: dict[str, dict[str, Any]] = {}
    for source in read_jsonl(args.sources):
        source_by_id[str(source["record_id"])] = source

    candidates: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    seen_ids: set[str] = set()
    for unit in read_jsonl(args.units):
        if unit.get("unit_type") != "heading_delimited_provisional_span" or unit.get("source_cycle_id") != "alf_layla_work_cycle":
            continue
        rid = str(unit["source_record_id"])
        if rid not in source_by_id:
            raise ValueError(f"No source provenance record for {rid}")
        cid = str(unit["unit_id"])
        if cid in seen_ids:
            raise ValueError(f"Duplicate story-unit id: {cid}")
        seen_ids.add(cid)
        selected_ids.add(rid)
        candidates.append(unit)
    if not candidates:
        raise ValueError("No Alf Layla provisional story units found")

    section_counts: dict[tuple[str, int, int], int] = {}
    for section in read_jsonl(args.sections):
        key = (str(section["source_record_id"]), int(section["start_char"]), int(section["end_char"]))
        if key in section_counts:
            raise ValueError(f"Duplicate heading-section offsets: {key}")
        section_counts[key] = int(section["direct_sentencepiece_tokens"])

    grouped: dict[str, list[dict[str, Any]]] = {}
    for rid in selected_ids:
        grouped[rid] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.source_zip) as archive:
        for rid in sorted(selected_ids):
            source = source_by_id[rid]
            if not normalize_heading(str(source.get("title", ""))).startswith("الف ليلة وليلة"):
                raise ValueError(f"Source record is outside the Alf Layla work: {rid}")
            if "CC BY 4.0" not in str(source.get("license", "")):
                raise ValueError(f"Source record has no explicit CC BY 4.0 provenance: {rid}")
            if not source.get("attribution"):
                raise ValueError(f"Missing source attribution: {rid}")
            match = re.search(r"hindawi-(\d+)$", rid)
            if not match:
                raise ValueError(f"Unexpected source id format: {rid}")
            member = f"corpus-txt/{match.group(1)}.txt"
            raw = archive.read(member)
            text = raw.decode("utf-8")
            actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if actual_hash != source["source_text_sha256"]:
                raise ValueError(f"Source SHA-256 mismatch for {rid}: {actual_hash}")

            rows = sorted((row for row in candidates if row["source_record_id"] == rid),
                          key=lambda row: int(row["char_start"]))
            previous_end = -1
            for row in rows:
                start, end = int(row["char_start"]), int(row["char_end"])
                if not (0 <= start < end <= len(text)):
                    raise ValueError(f"Invalid span offsets for {row['unit_id']}")
                if start < previous_end:
                    raise ValueError(f"Overlapping candidate spans in {rid}")
                previous_end = end
                section_key = (rid, start, end)
                if section_key not in section_counts:
                    raise ValueError(f"No tokenizer count for inventoried unit {row['unit_id']}")
                span = text[start:end]
                if normalize_heading(span.splitlines()[0] if span.splitlines() else "") != normalize_heading(row["title_candidate"]):
                    raise ValueError(f"Heading text does not match source offsets for {row['unit_id']}")
                item = {
                    "schema": "ansg.alf_layla_heading_story_candidate/1",
                    "candidate_id": row["unit_id"],
                    "title_candidate": row["title_candidate"],
                    "text": span,
                    "source": {
                        "record_id": rid,
                        "title": source.get("title"),
                        "author": source.get("author"),
                        "text_file": source.get("text_file"),
                        "source_text_sha256": actual_hash,
                        "license": source.get("license"),
                        "attribution": source.get("attribution"),
                        "dataset_doi": "10.5878/7rbh-gy93",
                        "archive_member": member,
                    },
                    "span": {
                        "start_char": start,
                        "end_char": end,
                        "boundary_start_evidence": row.get("boundary_start_evidence"),
                        "boundary_end_evidence": row.get("boundary_end_evidence"),
                        "boundary_method": "existing_provisional_inventory_offsets",
                        "source_candidate_status": row.get("status"),
                    },
                    "work_group": row.get("source_group_split_lock"),
                    "split": row.get("split"),
                    "candidate_scope": source.get("candidate_scope"),
                    "counts": {
                        "whitespace_words": len(span.split()),
                        "direct_sentencepiece_tokens": section_counts[section_key],
                    },
                    "status": "explicit_story_heading_span_requires_content_and_boundary_review",
                    "annotations": {"DIA": None, "MOTIF": None, "MORAL": None},
                    "review_flags": [
                        "confirm_title_heading_and_full_story_boundary",
                        "check_for_inner_tales_and_parent_resumption",
                        "do_not_treat_heading_as_human_annotation",
                    ],
                }
                grouped[rid].append(item)

    if args.max_bytes_per_file < 1024:
        parser.error("--max-bytes-per-file must be at least 1024")
    all_items = [item for rid in sorted(grouped) for item in grouped[rid]]
    shards: list[list[str]] = []
    lines: list[str] = []
    shard_bytes = 0
    for item in all_items:
        line = json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
        line_bytes = len(line.encode("utf-8"))
        if lines and shard_bytes + line_bytes > args.max_bytes_per_file:
            shards.append(lines)
            lines = []
            shard_bytes = 0
        lines.append(line)
        shard_bytes += line_bytes
    if lines:
        shards.append(lines)

    output_files: list[dict[str, Any]] = []
    for file_index, shard_lines in enumerate(shards, 1):
        path = args.output_dir / f"alf_layla_story_candidates_{file_index:03d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.writelines(shard_lines)
        output_files.append({
            "path": path.name,
            "records": len(shard_lines),
            "bytes": sum(len(line.encode("utf-8")) for line in shard_lines),
        })

    report = {
        "schema": "ansg.alf_layla_story_text_candidate_report/1",
        "status": "CC_BY_4.0_text_spans_for_review_not_verified_story_corpus",
        "dataset": "The Arabic E-Book Corpus Version 1",
        "dataset_doi": "10.5878/7rbh-gy93",
        "source_records": len(selected_ids),
        "source_records_with_candidates": sum(bool(v) for v in grouped.values()),
        "explicit_story_heading_candidates": len(all_items),
        "whitespace_words": sum(int(x["counts"]["whitespace_words"]) for x in all_items),
        "direct_sentencepiece_tokens": sum(int(x["counts"]["direct_sentencepiece_tokens"]) for x in all_items),
        "token_counts_source": "existing direct-build heading-section output using the pinned tokenizer model",
        "candidate_files": output_files,
        "source_text_hashes_verified": True,
        "span_offsets_matched_to_inventory": True,
        "all_candidates_share_work_group": len({x["work_group"] for x in all_items}) <= 1,
        "annotation_coverage": {"DIA": 0, "MOTIF": 0, "MORAL": 0, "triple_covered_tokens": 0},
        "limitations": [
            "Explicit story-title headings improve retrieval but do not prove the complete story boundary.",
            "The end offset is the next structural heading or source end; editorial headings may divide or combine narratives.",
            "All source volumes remain grouped together; these splits are not the thesis stratified split.",
            "The direct SentencePiece count is not the thesis root-aware tokenizer count.",
            "This is not the historical ANHC and does not reproduce the thesis's 13.3M, 23M or 70.4M figures.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
