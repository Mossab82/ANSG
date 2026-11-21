#!/usr/bin/env python3
"""Merge currently available ANHC text into a local, rights-gated staging view.

The output is quarantined and git-ignored because one input is a user-supplied
PDF whose edition and reuse rights are unresolved. Parallel Kalila versions are
linked by title only; no deduplication, story-boundary adjudication, annotation,
or split assignment is performed.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def title_key(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip()
    value = re.sub(r"^باب\s*", "", value)
    value = value.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"}))
    return "".join(ch for ch in value if ch.isalnum())


def source_record_id(record: dict[str, Any]) -> str:
    return str(record.get("narrative", {}).get("extra", {}).get("source_record_id") or record["record_id"])


def best_kalila_match(title: str, rows_by_title: list[tuple[str, dict[str, Any]]]) -> tuple[str | None, float]:
    key = title_key(title)
    scored = [
        (difflib.SequenceMatcher(None, key, existing_key).ratio(), source_id)
        for existing_key, source_id in rows_by_title
    ]
    if not scored:
        return None, 0.0
    score, source_id = max(scored)
    return (source_id, score) if score >= 0.88 else (None, score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--staging",
        type=Path,
        default=ROOT / "data/corpora/anhc/staging_records.jsonl",
    )
    parser.add_argument(
        "--pdf-candidates",
        type=Path,
        default=ROOT / "data/corpora/anhc/quarantine/kalila_pdf_candidates.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/corpora/anhc/quarantine/merged_narrative_candidates.jsonl",
    )
    parser.add_argument(
        "--source-units-output",
        type=Path,
        default=ROOT / "data/corpora/anhc/quarantine/non_narrative_source_units.jsonl",
    )
    parser.add_argument(
        "--review-queue-output",
        type=Path,
        default=ROOT / "data/corpora/anhc/quarantine/merged_story_review_queue.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/corpora/anhc/merged_candidate_manifest.json",
    )
    args = parser.parse_args()

    staged = read_jsonl(args.staging)
    pdf_rows = read_jsonl(args.pdf_candidates)
    kalila_title_index: list[tuple[str, str]] = []
    for record in staged:
        if record.get("source", {}).get("source_id", "").startswith("kalila_dimna-"):
            kalila_title_index.append((
                title_key(record.get("narrative", {}).get("title") or ""),
                source_record_id(record),
            ))

    merged: list[dict[str, Any]] = []
    source_units: list[dict[str, Any]] = []
    family_members: dict[str, list[str]] = defaultdict(list)
    links: list[dict[str, Any]] = []
    family_for_source: dict[str, str] = {}

    # Preserve all existing narrative candidates and group each Nights page/cycle
    # conservatively so nested stories cannot leak across future partitions.
    for original in staged:
        row = copy.deepcopy(original)
        src_id = source_record_id(row)
        family_id = f"family-candidate-{src_id}"
        family_for_source[src_id] = family_id
        row["narrative"].setdefault("extra", {})["candidate_story_family_id"] = family_id
        row["narrative"]["extra"]["family_grouping_status"] = "source_page_or_cycle_conservative_grouping"
        row["narrative"]["extra"]["merge_role"] = "existing_heritage_seed_candidate"
        row["narrative"]["extra"]["text_sha256"] = digest(row["text"])
        merged.append(row)
        family_members[family_id].append(row["record_id"])

    pdf_fable_count = 0
    pdf_frontmatter_count = 0
    title_match_scores: list[float] = []
    for candidate in pdf_rows:
        if candidate.get("unit_kind_candidate") == "front_matter":
            unit = copy.deepcopy(candidate)
            unit["source_unit_status"] = "retained_for_provenance_not_a_story_record"
            unit["text_sha256"] = digest(unit["text"])
            source_units.append(unit)
            pdf_frontmatter_count += 1
            continue

        pdf_fable_count += 1
        title = candidate["title_candidate"]
        matched, score = best_kalila_match(title, kalila_title_index)
        title_match_scores.append(score)
        family_id = family_for_source.get(matched) if matched else None
        if not family_id:
            family_id = f"family-candidate-pdf-{candidate['record_id']}"
        source = candidate["source"]
        text = candidate["text"]
        row = {
            "schema_version": "ansg.anhc-record/0.1",
            "record_id": candidate["record_id"],
            "text": text,
            "source": {
                "source_id": f"user-pdf-{source['filename']}",
                "title": source["filename"],
                "copyright_status": "unknown",
                "redistributable": False,
                "license": None,
                "edition": None,
                "editor": None,
                "publisher": None,
                "year": None,
                "url": None,
                "file_path": None,
                "file_sha256": source["sha256"],
                "notes": "User-supplied local PDF; exact edition and reuse rights are unknown. Text remains in git-ignored quarantine and is not cleared for training or redistribution.",
            },
            "narrative": {
                "collection": "ANHC local quarantined source candidates",
                "title": title,
                "story_index": None,
                "night_numbers": [],
                "parent_story_id": None,
                "nesting_level": None,
                "source_span": {"start": 0, "end": len(text)},
                "formulae": [],
                "extra": {
                    "unit_type": "fable_chapter_or_story_cycle_candidate",
                    "source_record_id": None,
                    "candidate_story_family_id": family_id,
                    "family_grouping_status": "title_based_parallel_source_candidate_needs_human_collation" if matched else "unmatched_pdf_title_candidate",
                    "parallel_candidate_source_record_id": matched,
                    "title_match_score": round(score, 4),
                    "source_pdf_page_start": source["pdf_page_start"],
                    "source_pdf_page_end": source["pdf_page_end"],
                    "printed_page_start": source["printed_page_start"],
                    "printed_page_end": source["printed_page_end"],
                    "source_pdf_sha256": source["sha256"],
                    "text_sha256": digest(text),
                    "segmentation_status": candidate["segmentation_status"],
                    "nested_story_status": "unreviewed",
                    "private_use_glyph_count": candidate["private_use_glyph_count"],
                    "local_quarantine_only": True,
                    "rights_and_training_gate": "blocked_pending_edition_and_rights_review",
                },
            },
            "dia": [],
            "motif": [],
            "moral": [],
            "annotation_status": {"dia": "unannotated", "motif": "unannotated", "moral": "unannotated"},
            "split": None,
            "normalization_config_sha256": None,
        }
        merged.append(row)
        family_members[family_id].append(row["record_id"])
        links.append({
            "pdf_record_id": row["record_id"],
            "pdf_title": title,
            "seed_source_record_id": matched,
            "candidate_story_family_id": family_id,
            "title_similarity": round(score, 4),
            "link_status": "title_only_needs_human_collation" if matched else "unmatched",
        })

    ids = [row["record_id"] for row in merged]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate record_id found while merging source candidates")

    write_jsonl(args.output, merged)
    write_jsonl(args.source_units_output, source_units)
    review_queue = []
    for row in merged:
        extra = row.get("narrative", {}).get("extra", {})
        review_queue.append({
            "record_id": row["record_id"],
            "candidate_story_family_id": extra.get("candidate_story_family_id"),
            "title_candidate": row.get("narrative", {}).get("title"),
            "source_id": row.get("source", {}).get("source_id"),
            "parallel_candidate_source_record_id": extra.get("parallel_candidate_source_record_id"),
            "source_pdf_pages": [extra.get("source_pdf_page_start"), extra.get("source_pdf_page_end")]
            if extra.get("source_pdf_page_start") is not None else None,
            "segmentation_status": extra.get("segmentation_status"),
            "boundary_review": None,
            "nested_story_review": None,
            "edition_collation_review": None,
            "rights_review": None,
            "split": None,
            "annotation_status": row.get("annotation_status", {}),
        })
    write_jsonl(args.review_queue_output, review_queue)
    word_counts = Counter()
    for row in merged:
        role = row.get("narrative", {}).get("extra", {}).get("merge_role")
        family = row.get("source", {}).get("source_id", "")
        if role == "existing_heritage_seed_candidate":
            key = "existing_heritage_seed_candidates"
        elif family.startswith("user-pdf-"):
            key = "user_pdf_parallel_candidates"
        else:
            key = "other_candidates"
        word_counts[key] += len(row["text"].split())

    report = {
        "schema": "ansg.anhc-merged-candidate-manifest/0.1",
        "dataset_id": "ANHC-reconstruction",
        "status": "merged_local_staging_review_required",
        "narrative_records": len(merged),
        "existing_staged_records": len(staged),
        "user_pdf_fable_candidates": pdf_fable_count,
        "front_matter_source_units_separate": pdf_frontmatter_count,
        "candidate_story_family_groups": len(family_members),
        "merged_story_review_queue_rows": len(review_queue),
        "kalila_title_links": sum(bool(link["seed_source_record_id"]) for link in links),
        "kalila_title_link_threshold": 0.88,
        "kalila_title_match_min": round(min(title_match_scores), 4) if title_match_scores else None,
        "kalila_title_match_max": round(max(title_match_scores), 4) if title_match_scores else None,
        "candidate_words_whitespace_including_parallel_versions": sum(word_counts.values()),
        "candidate_words_by_origin": dict(word_counts),
        "exact_text_duplicate_count": len(merged) - len({digest(row["text"]) for row in merged}),
        "unique_story_words_status": "not_computed; parallel editions and source cycles require human collation",
        "dialect_coverage": [],
        "annotation_status": {"dia": "unannotated", "motif": "unannotated", "moral": "unannotated"},
        "rights_cleared_for_training": False,
        "redistributable": False,
        "split_status": "unassigned",
        "split_leakage_checked": False,
        "narrative_boundary_review": "required; Nights and Kalila nested stories are not adjudicated",
        "human_collation_required": True,
        "next_phase_readiness": "review_queue_prepared; human structural review and versioned annotation guide are the next gates",
        "external_dialect_story_datasets_downloaded": False,
        "outputs": {
            "local_quarantined_records": str(args.output.relative_to(ROOT)),
            "local_nonstory_source_units": str(args.source_units_output.relative_to(ROOT)),
            "local_story_review_queue": str(args.review_queue_output.relative_to(ROOT)),
            "candidate_group_links": links,
        },
        "limitations": [
            "This merges available text candidates, not a completed or training-cleared ANHC.",
            "Kalila PDF text is local-only because its edition and reuse rights are unresolved.",
            "Kalila PDF and Wikisource records are linked by title only; they remain separate text versions pending human collation.",
            "The currently available narrative sources are classical Arabic; the planned dialect families are not yet represented by ingested stories.",
            "Counts are whitespace-word counts, include parallel editions, and are not tokenizer-specific corpus token counts.",
            "The thesis reference of 70.4M total and 23M annotated tokens is not met by this reconstruction.",
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "narrative_records": report["narrative_records"],
        "existing_staged_records": report["existing_staged_records"],
        "user_pdf_fable_candidates": report["user_pdf_fable_candidates"],
        "front_matter_source_units_separate": report["front_matter_source_units_separate"],
        "candidate_story_family_groups": report["candidate_story_family_groups"],
        "kalila_title_links": report["kalila_title_links"],
        "candidate_words_whitespace_including_parallel_versions": report["candidate_words_whitespace_including_parallel_versions"],
        "exact_text_duplicate_count": report["exact_text_duplicate_count"],
        "rights_cleared_for_training": report["rights_cleared_for_training"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
