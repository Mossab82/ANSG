#!/usr/bin/env python3
"""Extract auditable, provisional Arabic heading-bounded section candidates.

The script does not assert that a heading is a gold story boundary and never
creates DIA/MOTIF/MORAL labels. It emits Unicode character offsets into each
source record, retaining the already assigned source-level split.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


HEADING = re.compile(
    r"^\s*(?P<cue>الفصل|الباب|الحكاية|الحكايات|حكاية|حكايات|القصة|قصة|"
    r"المشهد|الجزء|الكتاب)\s*(?P<title>[^\n]{0,140})\s*$"
)


def read_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def read_split_map(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row["record_id"])
            if rid in result:
                raise ValueError(f"Duplicate split assignment for {rid} at {path}:{line_number}")
            result[rid] = {"work_group": str(row["work_group"]), "split": str(row["split"])}
    return result


def heading_matches(text: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    offset = 0
    for line_number, line in enumerate(text.splitlines(keepends=True), 1):
        content = line.rstrip("\r\n")
        if len(content.strip()) <= 160:
            match = HEADING.fullmatch(content)
            if match:
                matches.append({
                    "start_char": offset,
                    "heading_line": line_number,
                    "heading_text": content.strip(),
                    "cue": match.group("cue"),
                })
        offset += len(line)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, action="append", required=True,
                        help="Compressed source JSONL; may be repeated for train/validation/test.")
    parser.add_argument("--split-map", type=Path, required=True,
                        help="Builder split_assignments.jsonl; split groups are kept unchanged.")
    parser.add_argument("--tokenizer-model", type=Path, required=True,
                        help="SentencePiece model used by this direct build, not the thesis tokenizer.")
    parser.add_argument("--output", type=Path, required=True, help="Output provisional spans JSONL.")
    parser.add_argument("--report", type=Path, required=True, help="Output JSON summary.")
    parser.add_argument("--minimum-span-chars", type=int, default=250)
    args = parser.parse_args()

    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit("Install sentencepiece==0.2.0 to count direct-build tokens") from exc
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer_model))
    split_map = read_split_map(args.split_map)

    totals: Counter[str] = Counter()
    cue_counts: Counter[str] = Counter()
    seen_record_ids: set[str] = set()
    output_rows: list[dict[str, Any]] = []

    for corpus_path in args.corpus:
        for record in read_jsonl_gz(corpus_path):
            rid = str(record["record_id"])
            if rid in seen_record_ids:
                raise ValueError(f"Source record appears in more than one split: {rid}")
            seen_record_ids.add(rid)
            assignment = split_map.get(rid)
            if assignment is None:
                raise ValueError(f"No grouped split assignment found for source record {rid}")
            text = str(record.get("text", ""))
            actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            expected_hash = record.get("source_text_sha256")
            if expected_hash and actual_hash != expected_hash:
                raise ValueError(f"Source text hash mismatch for {rid}")

            totals["source_records"] += 1
            totals["source_sentencepiece_tokens"] += len(tokenizer.encode(text, out_type=int))
            scope = str(record.get("candidate_scope") or
                        record.get("narrative", {}).get("candidate_scope", "unspecified"))
            totals[f"records_scope_{scope}"] += 1
            matches = heading_matches(text)
            totals["records_with_heading_candidates"] += bool(matches)
            if not matches:
                totals["unheaded_source_sentencepiece_tokens"] += len(tokenizer.encode(text, out_type=int))
                continue

            prefix = text[:matches[0]["start_char"]]
            totals["unheaded_prefix_chars"] += len(prefix)
            totals["unheaded_prefix_whitespace_words"] += len(prefix.split())
            totals["unheaded_prefix_sentencepiece_tokens"] += len(tokenizer.encode(prefix, out_type=int))

            for index, heading in enumerate(matches):
                start = heading["start_char"]
                end = matches[index + 1]["start_char"] if index + 1 < len(matches) else len(text)
                span = text[start:end]
                token_count = len(tokenizer.encode(span, out_type=int))
                row = {
                    "schema": "ansg.heading_bounded_section_candidate/1",
                    "candidate_id": f"{rid}-section-{index + 1:04d}",
                    "source_record_id": rid,
                    "source_title": (record.get("metadata", {}).get("title") or
                                     record.get("source", {}).get("title")),
                    "candidate_scope": scope,
                    "source_text_sha256": actual_hash,
                    "start_char": start,
                    "end_char": end,
                    "heading_line": heading["heading_line"],
                    "heading_text": heading["heading_text"],
                    "boundary_cue": heading["cue"],
                    "observed_whitespace_words": len(span.split()),
                    "direct_sentencepiece_tokens": token_count,
                    "work_group": assignment["work_group"],
                    "split": assignment["split"],
                    "status": "provisional_heading_section_requires_content_and_boundary_review",
                    "annotations": {"DIA": None, "MOTIF": None, "MORAL": None},
                    "review_flags": ["confirm_this_section_is_narrative_or_story_content"] +
                                    (["span_below_minimum_length"]
                                     if len(span) < args.minimum_span_chars else []),
                }
                output_rows.append(row)
                totals["heading_bounded_candidates"] += 1
                totals["candidate_chars"] += len(span)
                totals["candidate_whitespace_words"] += row["observed_whitespace_words"]
                totals["candidate_sentencepiece_tokens"] += token_count
                totals[f"candidates_split_{assignment['split']}"] += 1
                cue_counts[heading["cue"]] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    report = {
        "schema": "ansg.heading_boundary_candidate_report/1",
        "status": "provisional_sections_not_gold_story_segmentation",
        "source_records_processed": len(seen_record_ids),
        "heading_cues": dict(sorted(cue_counts.items())),
        "counts": dict(totals),
        "candidate_token_coverage_of_input": round(
            totals["candidate_sentencepiece_tokens"] /
            max(totals["source_sentencepiece_tokens"], 1), 6
        ),
        "split_assignment_source": str(args.split_map),
        "source_texts_embedded_in_output": False,
        "annotation_coverage": {"DIA": 0, "MOTIF": 0, "MORAL": 0, "triple_covered_tokens": 0},
        "limitations": [
            "Structural Arabic heading matches identify sections, not confirmed story boundaries or narrative content.",
            "A chapter may contain multiple embedded stories; nesting and parent-child links are not inferred.",
            "Books without a matched heading and text before each book's first heading remain in the whole-book inventory and are not discarded.",
            "The source-level SHA-256 split is reused; this does not reproduce the thesis TF-IDF/DBSCAN split.",
            "Direct SentencePiece counts differ from the thesis root-aware tokenizer counts.",
            "No annotation labels are generated by this script.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_records_processed": len(seen_record_ids), **dict(totals)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
