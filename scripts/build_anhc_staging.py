#!/usr/bin/env python3
"""Convert the existing heritage-text seed into reviewable ANHC staging records.

This produces real text records with source spans and provenance; it does not
claim that title boundaries, rights, or annotations have been adjudicated.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data/corpora/classical_arabic/stories.jsonl"
OUT = ROOT / "data/corpora/anhc"
sys.path.insert(0, str(ROOT / "src"))
from ansg.data.nights import NightsConfig, segment_stories  # noqa: E402
from ansg.data.records import NarrativeMetadata, NarrativeRecord, SourceProvenance, Span, validate_record  # noqa: E402


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    source_rows = [json.loads(line) for line in SEED.read_text(encoding="utf-8").splitlines() if line.strip()]
    records: list[NarrativeRecord] = []
    review_queue: list[dict] = []
    counts = {"arabian_nights": 0, "kalila_dimna": 0}
    total_words = 0

    for source in source_rows:
        raw_text = source["text"]
        if sha(raw_text) != source["sha256"]:
            raise ValueError(f"Seed checksum mismatch: {source['id']}")
        family = source["corpus"]
        common_notes = (
            "Copied from the repository heritage seed. Copyright and exact edition/revision "
            "are not yet cleared for redistribution or model training."
        )
        provenance = SourceProvenance(
            source_id=source["id"], title=source["title"], copyright_status="unknown",
            redistributable=False, license=None, edition=None, url=source["url"],
            file_path="data/corpora/classical_arabic/stories.jsonl",
            file_sha256=source["sha256"], notes=common_notes,
        )

        if family == "arabian_nights":
            units, markers = segment_stories(raw_text, NightsConfig())
            if not units:
                raise ValueError(f"No candidate units found for {source['id']}")
            for unit in units:
                segment = raw_text[unit.start:unit.end]
                if not segment.strip():
                    continue
                rid = f"anhc-{source['id']}-{unit.index:03d}"
                title = unit.title or "مقدمة / إطار الحكاية (مرشح آلي)"
                rec = NarrativeRecord(
                    record_id=rid, text=segment, source=provenance,
                    narrative=NarrativeMetadata(
                        collection="ANHC classical Arabic staging",
                        title=title, story_index=unit.index,
                        night_numbers=unit.nights, parent_story_id=None,
                        nesting_level=None, source_span=Span(unit.start, unit.end),
                        formulae=unit.formulae,
                        extra={
                            "source_record_id": source["id"],
                            "unit_type": "candidate_tale_segment",
                            "segmentation_status": "automatic_title_boundary_review_required",
                            "boundary_method": "existing configurable regex in src/ansg/data/nights.py",
                            "source_revision_id": source.get("revision_id"),
                            "source_text_sha256": source["sha256"],
                            "night_heading_count_in_source": len(markers),
                            "night_membership_status": "automatic_candidate_review_required",
                        },
                    ), split=None,
                )
                records.append(rec)
                review_queue.append({
                    "record_id": rid, "source_record_id": source["id"],
                    "title_candidate": title, "start_char": unit.start,
                    "end_char": unit.end, "candidate_word_count": len(segment.split()),
                    "night_numbers_candidate": unit.nights,
                    "segmentation_status": "automatic_title_boundary_review_required",
                    "review_decision": None,
                })
                counts[family] += 1
                total_words += len(segment.split())
        elif family == "kalila_dimna":
            # Source pages have titled fable headings, but some contain nested tales.
            # Keep page boundaries as candidate units until human segmentation review.
            rid = f"anhc-{source['id']}-page"
            rec = NarrativeRecord(
                record_id=rid, text=raw_text, source=provenance,
                narrative=NarrativeMetadata(
                    collection="ANHC classical Arabic staging",
                    title=source["title"].split("/", 1)[-1],
                    source_span=Span(0, len(raw_text)),
                    extra={
                        "source_record_id": source["id"],
                        "unit_type": "titled_source_page_candidate",
                        "segmentation_status": "page_boundary_only_nested_tales_unreviewed",
                        "source_revision_id": source.get("revision_id"),
                        "source_text_sha256": source["sha256"],
                    },
                ), split=None,
            )
            records.append(rec)
            review_queue.append({
                "record_id": rid, "source_record_id": source["id"],
                "title_candidate": rec.narrative.title, "start_char": 0,
                "end_char": len(raw_text), "candidate_word_count": len(raw_text.split()),
                "segmentation_status": "page_boundary_only_nested_tales_unreviewed",
                "review_decision": None,
            })
            counts[family] += 1
            total_words += len(raw_text.split())
        else:
            raise ValueError(f"Unrecognized seed corpus: {family}")

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "staging_records.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    (OUT / "story_review_queue.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in review_queue),
        encoding="utf-8",
    )
    validation = [issue for rec in records for issue in validate_record(rec)]
    (OUT / "validation_report.json").write_text(json.dumps({
        "records_checked": len(records),
        "errors": sum(issue.severity == "error" for issue in validation),
        "warnings": sum(issue.severity == "warning" for issue in validation),
        "issues": [
            {"record_id": issue.record_id, "severity": issue.severity, "message": issue.message}
            for issue in validation
        ],
        "warning_policy": "Unknown source rights are expected at staging and block training admission.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema": "ansg.anhc-staging-manifest/0.1",
        "dataset_id": "ANHC-reconstruction",
        "dataset_status": "staging_records_populated_review_required",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(SEED.relative_to(ROOT)),
        "input_source_records": len(source_rows),
        "records": len(records),
        "records_by_family": counts,
        "candidate_words_whitespace_count": total_words,
        "token_count_status": "whitespace_word_count_only; tokenizer-specific token count not computed",
        "annotation_status": {"dia": "unannotated", "motif": "unannotated", "moral": "unannotated"},
        "rights_cleared_for_training": False,
        "redistributable": False,
        "split_status": "unassigned",
        "split_leakage_checked": False,
        "structural_review_queue": str((OUT / "story_review_queue.jsonl").relative_to(ROOT)),
        "validation_report": str((OUT / "validation_report.json").relative_to(ROOT)),
        "schema_validation_errors": sum(issue.severity == "error" for issue in validation),
        "schema_validation_warnings": sum(issue.severity == "warning" for issue in validation),
        "policy": "No automatic extraction or heuristic is represented as human annotation or gold data.",
        "counts_note": "ANHC is being constructed here; these are populated staging records, not the thesis target of 70.4M tokens / 23M annotated tokens.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("input_source_records", "records", "records_by_family", "candidate_words_whitespace_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
